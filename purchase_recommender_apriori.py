import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def load_data():
    """
    Load and preprocess the grocery transactions data.
    Returns a one-hot encoded DataFrame of transactions.
    """
    # Read the CSV file
    df = pd.read_csv('grocery_transactions.csv', header=None)
    
    # Clean and prepare transactions
    transactions = [[str(item).strip() for item in row if pd.notna(item) and str(item).strip()] 
                   for row in df.values]
    
    # Encode transactions into one-hot format
    te = TransactionEncoder()
    basket = pd.DataFrame(te.fit(transactions).transform(transactions), 
                         columns=te.columns_)
    return basket

def get_rules(basket):
    """
    Generate association rules with minimum 1% support and confidence.
    Returns a DataFrame of association rules.
    """
    # Find frequent itemsets
    frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
    
    # Generate rules with minimum 1% confidence
    rules = association_rules(frequent_itemsets, 
                            metric="confidence", 
                            min_threshold=0.01)
    return rules

def recommend(rules, items):
    """
    Recommend an item based on input items and association rules.
    Args:
        rules: DataFrame of association rules
        items: List of input items
    Returns:
        Recommended item or None if no recommendation found
    """
    # Clean and validate input items
    items = [i.strip().lower() for i in items if i.strip()]
    if not (1 <= len(items) <= 3):
        return None
    
    input_set = set(items)
    
    # Try to find rules matching the exact input set
    exact_matches = rules[rules['antecedents'].apply(
        lambda x: set(map(str.lower, x)) == input_set and len(x) == len(items)
    )]
    
    if not exact_matches.empty:
        # Get the rule with highest confidence
        best_rule = exact_matches.sort_values('confidence', ascending=False).iloc[0]
        for consequent in best_rule['consequents']:
            if consequent.lower() not in input_set:
                return consequent
    
    # If no exact match, try individual items
    for item in items:
        single_item_rules = rules[rules['antecedents'].apply(
            lambda x: set(map(str.lower, x)) == {item} and len(x) == 1
        )]
        
        if not single_item_rules.empty:
            best_rule = single_item_rules.sort_values('confidence', ascending=False).iloc[0]
            for consequent in best_rule['consequents']:
                if consequent.lower() not in input_set:
                    return consequent
    
    return None

def main():
    """
    Main function to run the purchase proposal system.
    """
    print("Load data and buil rules...")
    basket = load_data()
    rules = get_rules(basket)
    
    print("\nPurchase Proposal System")
    print("Enter 1-3 items (comma separated), or 'quit' to exit.")
    
    while True:
        # Get user input
        user_input = input("\nEnter items: ").strip().lower()
        
        # Check for quit command
        if user_input == 'quit':
            break
            
        # Check for empty input
        if not user_input:
            print("Please enter between 1 and 3 validitems.")
            continue
        
        # Process input items
        items = [item.strip() for item in user_input.split(',')]
        
        # Remove empty items from the list
        items = [item for item in items if item]
        
        # Check if we have any valid items after cleaning
        if not items:
            print("Please enter between 1 and 3 valid items.")
            continue
        
        # Validate input length
        if not (1 <= len(items) <= 3):
            print("Please enter between 1 and 3 valid items.")
            continue
        
        # Get and display recommendation
        recommendation = recommend(rules, items)
        if recommendation:
            print(f"Maybe you would also like to purchase {recommendation}")
        else:
            print("No recommendations available for these items.")

if __name__ == "__main__":
    main() 