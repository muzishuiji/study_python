from datetime import datetime, timedelta

def is_weekday(date):
    # 如果是周一到周五，返回True
    return date.weekday() < 5

def find_last_workday_of_month(from_date):
    # 计算下个月的第一天
    if from_date.month == 12:
        first_day_of_next_month = datetime(from_date.year + 1, 1, 1)
    else:
        first_day_of_next_month = datetime(from_date.year, from_date.month + 1, 1)
    # 当前月的最后一天是下个月第一天的前一天
    last_day_of_current_month = first_day_of_next_month - timedelta(days=1)

    # 如果最后一天是非工作日，向前查找直到找到工作日
    while not is_weekday(last_day_of_current_month):
        last_day_of_current_month -= timedelta(days=1)
    
    last_workday_end = datetime(last_day_of_current_month.year, last_day_of_current_month.month, last_day_of_current_month.day, 23, 59, 59)
    return last_workday_end

# 传入指定日期
def days_until_last_workday_of_month(from_date):
    # 查找最后一个工作日
    last_workday = find_last_workday_of_month(from_date)
    print(last_workday)
    # 计算与今天的差值
    remain_days = (last_workday - from_date).days
    return remain_days

target_date = datetime(2023, 12, 28)
print(days_until_last_workday_of_month(target_date))

def count_workdays_util_date(from_date):
     # 查找最后一个工作日
    last_workday = find_last_workday_of_month(from_date)
    # 初始化工作日计数
    working_days = 0
      # 计算从今天到月底的每一天
    for single_date in (from_date + timedelta(n) for n in range((last_workday - from_date).days + 1)):
        if is_weekday(single_date):
            working_days += 1
    return working_days

# 调用函数并打印结果
target_date = datetime(2024, 1, 15)
print(count_workdays_util_date(target_date))

