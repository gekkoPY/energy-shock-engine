# config.py
import os
from dotenv import load_dotenv

# Load local environment variables from .env file if present
load_dotenv()

# Federal Reserve Economic Data (FRED) API Key
# Automatically retrieves key from environment; falls back to placeholder for public repositories
FRED_API_KEY = os.getenv("FRED_API_KEY", "YOUR_FRED_API_KEY_HERE")

# Historical sample boundaries
START_DATE = "2007-05-10"
END_DATE = None  # None dynamically queries up to the most recent trading day

# High-frequency shock identification parameters
SHOCK_SIGMA_THRESHOLD = 1.5  # Multiplier for 60-day rolling Brent volatility

# Jordà (2005) Local Projections response horizons (in trading days)
LP_HORIZONS = [1, 3, 5, 10, 20, 60]
