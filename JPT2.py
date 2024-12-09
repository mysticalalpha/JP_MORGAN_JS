def calculate_contract_value(
    buy_price: float, 
    sell_price: float, 
    quantity: float, 
    storage_cost_per_month: float = 0, 
    storage_duration_months: int = 0, 
    injection_withdrawal_cost: float = 0, 
    transportation_cost: float = 0
) -> float:
    """
    Calculate the value of a trade agreement.

    Parameters:
    - buy_price (float): Price at which the commodity is bought ($/MMBtu).
    - sell_price (float): Price at which the commodity is sold ($/MMBtu).
    - quantity (float): Quantity of the commodity traded (in MMBtu).
    - storage_cost_per_month (float): Monthly storage cost ($).
    - storage_duration_months (int): Duration of storage (in months).
    - injection_withdrawal_cost (float): Injection/withdrawal cost per million MMBtu ($).
    - transportation_cost (float): Transportation cost for moving gas to/from facility ($).

    Returns:
    - float: The net value of the contract.
    """
   
    profit = (sell_price - buy_price) * quantity

   
    total_storage_cost = storage_cost_per_month * storage_duration_months
    total_injection_withdrawal_cost = injection_withdrawal_cost * (quantity / 1e6)  
    total_transportation_cost = transportation_cost * 2  

    
    net_value = profit - (total_storage_cost + total_injection_withdrawal_cost + total_transportation_cost)
    return round(net_value, 2)


buy_price = 2.0  
sell_price = 3.0 
quantity = 1e6  
storage_cost_per_month = 100000  
storage_duration_months = 4  
injection_withdrawal_cost = 10000  
transportation_cost = 50000  


contract_value = calculate_contract_value(
    buy_price,
    sell_price,
    quantity,
    storage_cost_per_month,
    storage_duration_months,
    injection_withdrawal_cost,
    transportation_cost
)

print(f"The value of the contract is: ${contract_value}")
