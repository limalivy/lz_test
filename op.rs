fn try_swap_optimized(
    &mut self,
    ctx: &OptContext,
    assignment: &mut [u8],
    r1: usize,
    r2: usize,
    temp: f64,
    rng: &mut ThreadRng,
) -> bool {
    let k1 = assignment[r1];
    let k2 = assignment[r2];
    
    // 快速过滤
    if k1 == k2 || 
       !ctx.dynamic_groups[r1].contains_key(k2) || 
       !ctx.dynamic_groups[r2].contains_key(k1) {
        return false;
    }

    // 缓存当前状态以便快速回滚
    let old_score = self.get_score(ctx);
    
    // 保存受影响汉字的状态（用于快速回滚）
    let affected1 = &ctx.group_to_char_indices[r1];
    let affected2 = &ctx.group_to_char_indices[r2];
    
    // 创建受影响汉字的唯一列表
    let mut affected_chars = Vec::with_capacity(affected1.len() + affected2.len());
    affected_chars.extend_from_slice(affected1);
    for &char_idx in affected2 {
        if !affected1.contains(&char_idx) {
            affected_chars.push(char_idx);
        }
    }
    
    let snapshot = self.create_swap_snapshot(&affected_chars);

    // 执行交换和增量更新
    assignment[r1] = k2;
    assignment[r2] = k1;
    self.update_swap_diff_fast(ctx, assignment, &affected_chars);

    let new_score = self.get_score(ctx);
    let delta = new_score - old_score;

    if delta <= 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
        true
    } else {
        // 快速回滚：恢复交换前的状态
        assignment[r1] = k1;
        assignment[r2] = k2;
        self.restore_swap_snapshot(snapshot);
        false
    }
}

#[inline(always)]
fn create_swap_snapshot(&self, affected_chars: &[usize]) -> SwapSnapshot {
    let mut snapshot = SwapSnapshot {
        codes: Vec::with_capacity(affected_chars.len()),
        keys: Vec::with_capacity(affected_chars.len()),
        equiv_contribs: Vec::with_capacity(affected_chars.len()),
        equiv_sq_contribs: Vec::with_capacity(affected_chars.len()),
        buckets: Vec::with_capacity(affected_chars.len() * 2), // 每个汉字最多影响2个桶
        bucket_freqs: Vec::with_capacity(affected_chars.len() * 2),
        collision_count: self.total_collisions,
        collision_frequency: self.collision_frequency,
        total_equiv_weighted: self.total_equiv_weighted,
        total_equiv_sq_weighted: self.total_equiv_sq_weighted,
        key_weighted_usage: self.key_weighted_usage,
    };
    
    for &char_idx in affected_chars {
        // 保存每个受影响汉字的状态
        snapshot.codes.push((char_idx, self.current_codes[char_idx]));
        snapshot.keys.push((char_idx, self.current_keys[char_idx]));
        snapshot.equiv_contribs.push((char_idx, self.current_equiv_contrib[char_idx]));
        snapshot.equiv_sq_contribs.push((char_idx, self.current_equiv_sq_contrib[char_idx]));
        
        // 保存相关桶的状态
        let code = self.current_codes[char_idx];
        snapshot.buckets.push((code, self.buckets[code]));
        snapshot.bucket_freqs.push((code, self.bucket_freqs[code]));
    }
    
    snapshot
}

#[inline(always)]
fn restore_swap_snapshot(&mut self, snapshot: SwapSnapshot) {
    // 恢复全局状态
    self.total_collisions = snapshot.collision_count;
    self.collision_frequency = snapshot.collision_frequency;
    self.total_equiv_weighted = snapshot.total_equiv_weighted;
    self.total_equiv_sq_weighted = snapshot.total_equiv_sq_weighted;
    self.key_weighted_usage.copy_from_slice(&snapshot.key_weighted_usage);
    
    // 恢复每个汉字的状态
    for (char_idx, code) in snapshot.codes {
        self.current_codes[char_idx] = code;
    }
    for (char_idx, keys) in snapshot.keys {
        self.current_keys[char_idx] = keys;
    }
    for (char_idx, equiv) in snapshot.equiv_contribs {
        self.current_equiv_contrib[char_idx] = equiv;
    }
    for (char_idx, equiv_sq) in snapshot.equiv_sq_contribs {
        self.current_equiv_sq_contrib[char_idx] = equiv_sq;
    }
    
    // 恢复桶状态
    for (code, count) in snapshot.buckets {
        self.buckets[code] = count;
    }
    for (code, freq) in snapshot.bucket_freqs {
        self.bucket_freqs[code] = freq;
    }
}

#[inline(always)]
fn update_swap_diff_fast(&mut self, ctx: &OptContext, assignment: &[u8], affected_chars: &[usize]) {
    // 使用局部变量减少全局状态访问
    let mut local_collisions = self.total_collisions;
    let mut local_collision_freq = self.collision_frequency;
    let mut local_equiv_weighted = self.total_equiv_weighted;
    let mut local_equiv_sq_weighted = self.total_equiv_sq_weighted;
    let mut local_key_usage = self.key_weighted_usage;
    
    for &char_idx in affected_chars {
        let old_code = self.current_codes[char_idx];
        let (old_keys, old_num_keys) = self.current_keys[char_idx];
        
        let (new_code, new_keys, new_num_keys) = ctx.calc_code_and_keys(char_idx, assignment);

        if old_code == new_code {
            continue;
        }

        let freq = ctx.char_infos[char_idx].frequency;
        let freq_f = freq as f64;

        // 更新桶状态
        let (collision_delta, freq_delta) = self.update_buckets_for_char(old_code, new_code, freq);
        local_collisions = (local_collisions as isize + collision_delta) as usize;
        local_collision_freq = (local_collision_freq as i64 + freq_delta) as u64;

        // 更新当量统计
        let old_contrib = self.current_equiv_contrib[char_idx];
        let old_sq_contrib = self.current_equiv_sq_contrib[char_idx];
        
        let new_key_avg_equiv = ctx.calc_key_avg_equiv_inline(new_keys, new_num_keys);
        let new_contrib = new_key_avg_equiv * freq_f;
        let new_sq_contrib = new_key_avg_equiv * new_key_avg_equiv * freq_f;

        local_equiv_weighted += new_contrib - old_contrib;
        local_equiv_sq_weighted += new_sq_contrib - old_sq_contrib;
        
        self.current_equiv_contrib[char_idx] = new_contrib;
        self.current_equiv_sq_contrib[char_idx] = new_sq_contrib;

        // 更新按键分布
        let old_n = old_num_keys as usize;
        let new_n = new_num_keys as usize;
        
        for j in 0..old_n {
            let key_idx = old_keys[j] as usize;
            local_key_usage[key_idx] -= freq_f;
        }
        for j in 0..new_n {
            let key_idx = new_keys[j] as usize;
            local_key_usage[key_idx] += freq_f;
        }

        // 更新缓存
        self.current_codes[char_idx] = new_code;
        self.current_keys[char_idx] = (new_keys, new_num_keys);
    }
    
    // 批量写回全局状态
    self.total_collisions = local_collisions;
    self.collision_frequency = local_collision_freq;
    self.total_equiv_weighted = local_equiv_weighted;
    self.total_equiv_sq_weighted = local_equiv_sq_weighted;
    self.key_weighted_usage = local_key_usage;
}

#[inline(always)]
fn update_buckets_for_char(&mut self, old_code: usize, new_code: usize, freq: u64) -> (isize, i64) {
    let mut collision_delta = 0isize;
    let mut freq_delta = 0i64;
    
    // 处理旧编码
    let old_count = self.buckets[old_code];
    if old_count > 1 {
        collision_delta -= 1;
        freq_delta -= freq as i64;
        if old_count == 2 {
            freq_delta -= (self.bucket_freqs[old_code] - freq) as i64;
        }
    }
    self.buckets[old_code] = old_count.wrapping_sub(1);
    self.bucket_freqs[old_code] -= freq;

    // 处理新编码
    let new_count = self.buckets[new_code];
    if new_count >= 1 {
        collision_delta += 1;
        freq_delta += freq as i64;
        if new_count == 1 {
            freq_delta += self.bucket_freqs[new_code] as i64;
        }
    }
    self.buckets[new_code] = new_count.wrapping_add(1);
    self.bucket_freqs[new_code] += freq;
    
    (collision_delta, freq_delta)
}

#[derive(Default, Clone)]
struct SwapSnapshot {
    codes: Vec<(usize, usize)>,
    keys: Vec<(usize, ([u8; MAX_PARTS], u8))>,
    equiv_contribs: Vec<(usize, f64)>,
    equiv_sq_contribs: Vec<(usize, f64)>,
    buckets: Vec<(usize, u16)>,
    bucket_freqs: Vec<(usize, u64)>,
    collision_count: usize,
    collision_frequency: u64,
    total_equiv_weighted: f64,
    total_equiv_sq_weighted: f64,
    key_weighted_usage: [f64; EQUIV_TABLE_SIZE],
}
