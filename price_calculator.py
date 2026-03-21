import math

def get_decimal(value):
    """
    获取数值的小数部分
    
    Parameters:
    value (float): 输入的数值
    
    Returns:
    float: 小数部分
    """
    return round(value, 1) - math.floor(value)

def adjust_price(value):
    """
    根据小数部分进行价格调整
    
    Parameters:
    value (float): 输入的数值
    
    Returns:
    float: 调整后的价格
    """
    base_value = math.floor(value)
    dec = get_decimal(value)

    if dec == 0:
        return base_value
    elif 0 < dec <= 0.5:
        return base_value + 0.5
    else:
        return base_value + 0.9

def spec_adjustment(value):
    """
    根据个位数字进行特殊调整
    
    Parameters:
    value (float): 输入的数值
    
    Returns:
    float: 调整后的价格
    """
    last_digit = int(str(int(value))[-1])

    if last_digit in [4, 7, 1]:
        return value + 1
    return value

def AutoSalePrice(MyValue):
    """
    根据输入的价格进行自动定价
    
    Parameters:
    MyValue (float): 输入的价格
    
    Returns:
    float: 定价后的价格
    """
    if MyValue < 5:
        return spec_adjustment(round(MyValue, 1) * 10) / 10

    elif 5 <= MyValue < 30:
        adjusted_value = adjust_price(MyValue)
        adjusted_value = spec_adjustment(adjusted_value * 10) / 10

        if int(str(int(adjusted_value * 10))[-1]) == 0:
            adjusted_value -= 0.1

        return adjusted_value

    else:
        base_value = round(MyValue, 0)
        adjusted_value = spec_adjustment(base_value)

        if int(str(int(adjusted_value))[-1]) == 0:
            adjusted_value -= 0.1
        elif int(str(int(adjusted_value))[-1]) == 1:
            adjusted_value -= 1.1

        return adjusted_value

    # 根据最终价格进行额外调整
    if 60 < adjusted_value < 100:
        return math.floor(adjusted_value)
    
    if adjusted_value >= 100:
        last_digit = int(str(int(adjusted_value))[-1])
        if last_digit == 0:
            return adjusted_value - 2
        elif 1 <= last_digit < 5:
            return math.floor(adjusted_value / 10) * 10 + 5
        else:
            return math.floor(adjusted_value / 10) * 10 + 8
