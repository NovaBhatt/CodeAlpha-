import requests

class Stock:
    def __init__(self, symbol, name, quantity, purchase_price):
        # This is the constructor for the Stock class. It initializes the stock with a symbol (like AAPL for Apple),
        # a name (like Apple Inc.), the quantity of shares you own, and the price you bought them at.
        self.symbol = symbol
        self.name = name
        self.quantity = quantity
        self.purchase_price = purchase_price
        self.current_price = None  # This will store the current price of the stock, but we don't know it yet.

    def update_current_price(self, price):
        # This method updates the current price of the stock. We need this to know how much our stocks are worth now.
        self.current_price = price

    def calculate_stock_value(self):
        # This method calculates the total value of the stock by multiplying the quantity of shares by the current price.
        # For example, if you have 10 shares and each is worth $100, the total value is $1000.
        return self.quantity * self.current_price

class Portfolio:
    def __init__(self):
        # This is the constructor for the Portfolio class. It initializes an empty list to hold all the stocks we own.
        self.stocks = []

    def add_stock(self, stock):
        # This method adds a stock to our portfolio. We just append the stock to our list of stocks.
        self.stocks.append(stock)

    def remove_stock(self, stock):
        # This method removes a stock from our portfolio. We remove the stock from our list of stocks.
        self.stocks.remove(stock)

    def update_stock_price(self, symbol, price):
        # This method updates the price of a stock in our portfolio. We loop through all our stocks to find the one with the matching symbol.
        # Once we find it, we update its price and print a message. If we don't find it, we print an error message.
        for stock in self.stocks:
            if stock.symbol == symbol:
                stock.update_current_price(price)
                print(f"Updated {stock.name} ({stock.symbol}) current price to ${price:.2f}")
                break
        else:
            print(f"Stock with symbol {symbol} not found in portfolio.")

    def calculate_portfolio_value(self):
        # This method calculates the total value of our portfolio by summing up the value of all our stocks.
        # We loop through all our stocks and add up their values. If a stock's current price is None, we skip it.
        total_value = 0
        for stock in self.stocks:
            if stock.current_price is not None:
                total_value += stock.calculate_stock_value()
        return total_value

    def fetch_real_time_price(self, symbol):
        # This method fetches the real-time price of a stock using the Alpha Vantage API.
        # We build the URL with our API key and the stock symbol, then make a request to the API.
        # We parse the JSON response to get the price. If there's an error, we print a message and return None.
        api_key = 'YOUR_API_KEY'  # Replace with your Alpha Vantage API key
        url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}'
        response = requests.get(url)
        data = response.json()
        try:
            price = float(data['Global Quote']['05. price'])
            return price
        except KeyError:
            print(f"Error fetching price for {symbol}")
            return None

if __name__ == "__main__":
    my_portfolio = Portfolio()

    while True:
        # This is the main loop of our program. It displays a menu and lets the user choose an option.
        # Depending on the choice, we add a stock, remove a stock, calculate the portfolio value, fetch a real-time price, or exit the program.
        print("\nStock Portfolio Tracker")
        print("1. Add Stock")
        print("2. Remove Stock")
        print("3. Calculate Portfolio Value")
        print("4. Fetch Real-Time Price")
        print("5. Exit")

        choice = input("Enter your choice (1/2/3/4/5): ")

        if choice == "1":
            # If the user chooses to add a stock, we ask for the stock symbol, name, and quantity.
            # We fetch the real-time price and calculate the purchase price. Then we create a Stock object and add it to the portfolio.
            symbol = input("Enter stock symbol: ")
            name = input("Enter stock name: ")
            quantity = float(input("Enter quantity: "))
            purchase_price = my_portfolio.fetch_real_time_price(symbol) * quantity
            if purchase_price is not None:
                stock = Stock(symbol, name, quantity, purchase_price)
                print("Stock: ",stock.quantity)
                my_portfolio.add_stock(stock)
                print(f"{name} ({symbol}) added to portfolio at purchase price ${purchase_price:.2f}.")
        elif choice == "2":
            # If the user chooses to remove a stock, we ask for the stock symbol.
            # We loop through the portfolio to find the stock and remove it. If we don't find it, we print an error message.
            symbol = input("Enter stock symbol to remove: ")
            for stock in my_portfolio.stocks:
                if stock.symbol == symbol:
                    my_portfolio.remove_stock(stock)
                    print(f"{stock.name} ({stock.symbol}) removed from portfolio.")
                    break
            else:
                print(f"Stock with symbol {symbol} not found in portfolio.")
        elif choice == "3":
            # If the user chooses to calculate the portfolio value, we call the method to calculate it and print the result.
            total_value = my_portfolio.calculate_portfolio_value()
            print(f"Total portfolio value: ${total_value:.2f}")
        elif choice == "4":
            # If the user chooses to fetch a real-time price, we ask for the stock symbol.
            # We fetch the price and update the stock's price in the portfolio.
            symbol = input("Enter stock symbol to fetch real-time price: ")
            price = my_portfolio.fetch_real_time_price(symbol)
            if price is not None:
                my_portfolio.update_stock_price(symbol, price)
        elif choice == "5":
            # If the user chooses to exit, we break out of the loop and end the program.
            print("Exiting. Have a great day!")
            break
        else:
            # If the user enters an invalid choice, we print an error message.
            print("Invalid choice. Please select 1, 2, 3, 4, or 5.")
