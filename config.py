from dotenv import load_dotenv
import os

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Model to use for memo generation
ANTHROPIC_MODEL = "claude-sonnet-4-6"

# Number of trading days to look back for risk calculations
LOOKBACK_DAYS = 90

# Number of top holdings (by weight) to fetch news for
TOP_N_NEWS = 5

# Concentration warning threshold — flag any single holding above this weight
CONCENTRATION_THRESHOLD = 0.20

# Benchmark ticker for beta calculation
BENCHMARK_TICKER = "SPY"
