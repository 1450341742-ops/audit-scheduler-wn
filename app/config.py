TRAVEL_BUFFER_DAYS = 1
GAP_THRESHOLD_DAYS = 3
CHAIN_GAP_DAYS = 1
CHAIN_DISTANCE_KM = 300.0
CHAIN_BONUS = 6
EXPERT_MEMBERS_MUST_BE_A = False
DEFAULT_DISTANCE_KM = 800.0

# 批量排班优化模式权重（越大越重视该项）
OPT_ALPHA_DISTANCE = 0.8   # 总距离权重
OPT_BETA_LOAD = 1.2       # 负荷均衡权重（月度院次偏离均值惩罚）
OPT_GAMMA_WEEK = 3.0      # 本周任务数惩罚（更强避免集中）
OPT_DELTA_CONT = 0.8      # 连续出差惩罚

# 软指定专家/老师加分（当任务填写 preferred_experts 时，命中姓名则加分）
EXPERT_PREFERENCE_BONUS = 18

# 距离优先权重（越大越重视节省差旅）
DIST_PENALTY_DIVISOR = 35.0   # km / divisor 作为扣分，divisor越小距离扣分越大
NEAR_BONUS_0_100 = 22
NEAR_BONUS_100_300 = 16
NEAR_BONUS_300_800 = 8
NEAR_BONUS_GT_800 = 0
LOAD_PENALTY_FACTOR = 0.6     # 负荷惩罚系数（越小越不看负荷）

# 自动距离：若城市距离表未命中，则尝试用城市经纬度计算直线距离（并缓存到 CityDistance）
AUTO_DISTANCE_BY_GEO = True
# 若城市坐标缺失：是否允许使用默认距离兜底
FALLBACK_TO_DEFAULT_DISTANCE_IF_NO_GEO = True
