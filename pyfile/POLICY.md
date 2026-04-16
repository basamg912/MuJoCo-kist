# (수정 필요) Observation dim
dif_torso_com -> WL3 body 에 대해서만 적용되도록 변경 -> 96차원 -> 15차원

0~2    : base_ang_vel × 0.2       (3)
3~5    : projected_gravity         (3)
6~8    : velocity_commands         (3)
9~22   : joint_pos_rel (다리 14개) (14)
23~36  : joint_vel_rel × 0.05      (14)
37~67  : last_action               (31)
68~163 : dif_torso_com (32 body)   (96)