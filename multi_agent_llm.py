import ollama
import sqlite3
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression

# ========= SETUP =========
#Problem Statement 1: Optimizing Retail Inventory with Multi Agents

conn = sqlite3.connect('supply_chain.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS agent_logs (
    agent TEXT,
    task TEXT,
    response TEXT
)
''')
conn.commit()

# ========= COMMON TOOLS =========
def run_ollama(prompt, model_name="mistral"):
    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def web_scrape(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()[:500]

def embed_text(text, model="nomic-embed-text"):
    result = ollama.embeddings(model=model, prompt=text)
    return result['embedding']

def log_agent(agent, task, response):
    cursor.execute("INSERT INTO agent_logs VALUES (?, ?, ?)", (agent, task, response))
    conn.commit()

# ========= AGENT BASE CLASS =========
class Agent:
    def __init__(self, name):
        self.name = name

    def run(self):
        raise NotImplementedError("Each agent must implement a run method.")

# ========= INDIVIDUALAGENTS =========
class PricingAgent(Agent):
    def __init__(self, df):
        super().__init__("PricingAgent")
        self.df = df

    def run(self):
        if 'Price' not in self.df.columns or 'Sales Volume' not in self.df.columns:
            raise ValueError("CSV must contain 'Price' and 'Sales Volume' columns.")
        X = self.df[['Price']]
        y = self.df['Sales Volume']
        model = LinearRegression().fit(X, y)
        predicted = model.predict([[20]])[0]
        response = f"Predicted sales at price 20: {predicted:.2f}"
        log_agent(self.name, "Optimize pricing", response)
        return response

class DemandAgent(Agent):
    def __init__(self, df):
        super().__init__("DemandAgent")
        self.df = df

    def run(self):
        summary = self.df.describe().to_string()
        response = run_ollama(f"Analyze this demand data:\n{summary}")
        log_agent(self.name, "Demand Forecasting", response)
        return response

class InventoryAgent(Agent):
    def __init__(self, df):
        super().__init__("InventoryAgent")
        self.df = df

    def run(self):
        print("InventoryAgent running. Columns available:", self.df.columns.tolist())

        if 'Stock Levels' in self.df.columns:
            col = 'Stock Levels'
        else:
            raise ValueError("Expected column 'Stock Levels' not found in inventory dataset.")

        low_stock = self.df[self.df[col] < 50]
        response = f"Low stock items:\n{low_stock.to_string(index=False)}"
        log_agent(self.name, "Inventory Monitoring", response)
        return response



# ========= COORDINATOR =========
class MultiAgentCoordinator:
    def __init__(self, agents):
        self.agents = agents

    def run_all(self):
        for agent in self.agents:
            print(f"\nRunning {agent.name}...")
            print(agent.run())

# ========= MAIN =========
if __name__ == "__main__":
    pricing_df = pd.read_csv("C:/Users/sanja/OneDrive/Desktop/RHP/3 days session/Nodejs/accenture/pricing_optimization.csv")
    demand_df = pd.read_csv("C:/Users/sanja/OneDrive/Desktop/RHP/3 days session/Nodejs/accenture/demand_forecasting.csv")
    inventory_df = pd.read_csv("C:/Users/sanja/OneDrive/Desktop/RHP/3 days session/Nodejs/accenture/inventory_monitoring.csv")  

    agents = [
        PricingAgent(pricing_df),
        DemandAgent(demand_df),
        InventoryAgent(inventory_df)
    ]

    coordinator = MultiAgentCoordinator(agents)
    coordinator.run_all()

    print("\n✅ Agent logs stored in supply_chain.db")
