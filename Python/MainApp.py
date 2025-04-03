import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import subprocess
import threading
import os
import tempfile
import json
from datetime import datetime, timedelta

# Wrapper to interact with the C++ simulator
class CppSimulatorWrapper:
    def __init__(self, cpp_executable_path="./hft_simulator"):
        self.cpp_executable_path = cpp_executable_path
        self.output_dir = tempfile.mkdtemp()
        self.process = None
        self.is_running = False
        self.current_step = 0
        self.performance_data = None
        self.market_data = None

    def generate_sample_data(self, days=1, interval_seconds=1):
        """Generate sample tick data for testing"""
        # Create a date range
        start_date = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
        end_date = start_date + timedelta(days=days)

        # Generate timestamps
        timestamps = []
        current = start_date
        while current < end_date:
            if 9 <= current.hour < 16 or (current.hour == 9 and current.minute >= 30):
                timestamps.append(current)
            current += timedelta(seconds=interval_seconds)

        # Base price and random walk
        base_price = 100.0
        num_points = len(timestamps)
        price_changes = np.random.normal(0, 0.0005, num_points)  # Smaller changes for HFT

        # Calculate prices
        prices = []
        current_price = base_price
        for change in price_changes:
            current_price *= (1 + change)
            current_price = max(current_price, 0.01)
            prices.append(round(current_price, 2))

        # Generate volumes
        volumes = np.random.randint(100, 1000, num_points)

        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes
        })

        # Save to CSV for C++ simulator
        data_path = os.path.join(self.output_dir, "tick_data.csv")
        df.to_csv(data_path, index=False)
        return data_path

    def start_simulation(self, strategy_params=None, data_path=None):
        """Start the C++ simulator process"""
        if self.is_running:
            self.stop_simulation()

        if data_path is None:
            data_path = self.generate_sample_data()

        # Create params file
        params_path = os.path.join(self.output_dir, "params.json")
        if strategy_params is None:
            strategy_params = {
                "strategy": "MarketMaking",
                "spread_bps": 2.0,
                "max_position": 1000,
                "order_size": 100,
                "order_expiry": 5
            }

        with open(params_path, 'w') as f:
            json.dump(strategy_params, f)

        # Output paths
        perf_path = os.path.join(self.output_dir, "performance.csv")
        market_path = os.path.join(self.output_dir, "market_data.csv")

        # Start C++ process
        cmd = [
            self.cpp_executable_path,
            "--data", data_path,
            "--params", params_path,
            "--perf-output", perf_path,
            "--market-output", market_path,
            "--stream-mode"  # Enable streaming output for real-time visualization
        ]

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        self.is_running = True
        self.current_step = 0

        # Start thread to read output
        self.reader_thread = threading.Thread(target=self._read_output)
        self.reader_thread.daemon = True
        self.reader_thread.start()

        return True

    def _read_output(self):
        """Read output from the C++ process"""
        while self.is_running and self.process and self.process.poll() is None:
            line = self.process.stdout.readline().strip()
            if line:
                try:
                    data = json.loads(line)
                    if 'step' in data:
                        self.current_step = data['step']
                    # Process other status updates as needed
                except json.JSONDecodeError:
                    pass
            time.sleep(0.01)

    def stop_simulation(self):
        """Stop the simulation"""
        if self.process:
            self.process.terminate()
            self.process = None
        self.is_running = False

    def get_performance_data(self):
        """Get current performance data"""
        perf_path = os.path.join(self.output_dir, "performance.csv")
        if os.path.exists(perf_path):
            try:
                df = pd.read_csv(perf_path)
                self.performance_data = df
                return df
            except Exception as e:
                print(f"Error reading performance data: {e}")
        return None

    def get_market_data(self):
        """Get current market data"""
        market_path = os.path.join(self.output_dir, "market_data.csv")
        if os.path.exists(market_path):
            try:
                df = pd.read_csv(market_path)
                self.market_data = df
                return df
            except Exception as e:
                print(f"Error reading market data: {e}")
        return None

    def get_current_state(self):
        """Get current simulation state"""
        return {
            "is_running": self.is_running,
            "current_step": self.current_step,
            "performance_data": self.get_performance_data(),
            "market_data": self.get_market_data()
        }

# Initialize simulator
simulator = CppSimulatorWrapper()

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "HFT Simulator Dashboard"

# Define layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("HFT Simulator Dashboard", className="text-center my-4"),
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Simulation Controls"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Strategy Type"),
                            dcc.Dropdown(
                                id="strategy-dropdown",
                                options=[
                                    {"label": "Market Making", "value": "MarketMaking"},
                                    {"label": "Momentum", "value": "Momentum"},
                                    {"label": "Mean Reversion", "value": "MeanReversion"}
                                ],
                                value="MarketMaking"
                            )
                        ], width=4),

                        dbc.Col([
                            html.Label("Spread (bps)"),
                            dbc.Input(id="spread-input", type="number", value=2, min=0.5, max=10, step=0.5)
                        ], width=2),

                        dbc.Col([
                            html.Label("Max Position"),
                            dbc.Input(id="max-position-input", type="number", value=1000, min=100, max=10000, step=100)
                        ], width=2),

                        dbc.Col([
                            html.Label("Order Size"),
                            dbc.Input(id="order-size-input", type="number", value=100, min=10, max=1000, step=10)
                        ], width=2),

                        dbc.Col([
                            html.Label("Order Expiry (ticks)"),
                            dbc.Input(id="order-expiry-input", type="number", value=5, min=1, max=20, step=1)
                        ], width=2)
                    ]),

                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Start Simulation", id="start-button", color="success", className="mt-3 me-2"),
                            dbc.Button("Stop Simulation", id="stop-button", color="danger", className="mt-3"),
                        ], width=12)
                    ])
                ])
            ]),
        ], width=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Performance Metrics"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("P&L", className="card-title"),
                                    html.H3(id="pnl-value", children="$0.00")
                                ])
                            ])
                        ], width=3),

                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Sharpe Ratio", className="card-title"),
                                    html.H3(id="sharpe-value", children="0.00")
                                ])
                            ])
                        ], width=3),

                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Max Drawdown", className="card-title"),
                                    html.H3(id="drawdown-value", children="0.00%")
                                ])
                            ])
                        ], width=3),

                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Trades", className="card-title"),
                                    html.H3(id="trades-value", children="0")
                                ])
                            ])
                        ], width=3)
                    ])
                ])
            ])
        ], width=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("P&L Chart"),
                dbc.CardBody([
                    dcc.Graph(id="pnl-chart")
                ])
            ])
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Position Chart"),
                dbc.CardBody([
                    dcc.Graph(id="position-chart")
                ])
            ])
        ], width=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Order Book Heatmap"),
                dbc.CardBody([
                    dcc.Graph(id="orderbook-heatmap")
                ])
            ])
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Price Chart"),
                dbc.CardBody([
                    dcc.Graph(id="price-chart")
                ])
            ])
        ], width=6)
    ]),

    dcc.Interval(
        id='interval-component',
        interval=1000,  # in milliseconds
        n_intervals=0
    )
], fluid=True)

# Callbacks
@app.callback(
    Output("start-button", "disabled"),
    Output("stop-button", "disabled"),
    Input("interval-component", "n_intervals")
)
def update_button_state(n):
    return simulator.is_running, not simulator.is_running

@app.callback(
    Output("start-button", "n_clicks"),
    Input("start-button", "n_clicks"),
    State("strategy-dropdown", "value"),
    State("spread-input", "value"),
    State("max-position-input", "value"),
    State("order-size-input", "value"),
    State("order-expiry-input", "value"),
    prevent_initial_call=True
)
def start_simulation(n_clicks, strategy, spread, max_position, order_size, order_expiry):
    if n_clicks:
        params = {
            "strategy": strategy,
            "spread_bps": float(spread),
            "max_position": int(max_position),
            "order_size": int(order_size),
            "order_expiry": int(order_expiry)
        }
        simulator.start_simulation(params)
    return None

@app.callback(
    Output("stop-button", "n_clicks"),
    Input("stop-button", "n_clicks"),
    prevent_initial_call=True
)
def stop_simulation(n_clicks):
    if n_clicks:
        simulator.stop_simulation()
    return None

@app.callback(
    [
        Output("pnl-value", "children"),
        Output("sharpe-value", "children"),
        Output("drawdown-value", "children"),
        Output("trades-value", "children"),
        Output("pnl-chart", "figure"),
        Output("position-chart", "figure"),
        Output("orderbook-heatmap", "figure"),
        Output("price-chart", "figure")
    ],
    Input("interval-component", "n_intervals")
)
def update_charts(n):
    # Default values
    pnl_value = "$0.00"
    sharpe_value = "0.00"
    drawdown_value = "0.00%"
    trades_value = "0"

    # Default figures
    pnl_fig = go.Figure()
    position_fig = go.Figure()
    orderbook_fig = go.Figure()
    price_fig = go.Figure()

    # Update with real data if available
    perf_data = simulator.get_performance_data()
    market_data = simulator.get_market_data()

    if perf_data is not None and not perf_data.empty:
        # Update metric values
        if 'PnL' in perf_data.columns:
            latest_pnl = perf_data['PnL'].iloc[-1] - perf_data['PnL'].iloc[0]
            pnl_value = f"${latest_pnl:.2f}"

        # Calculate Sharpe ratio if we have enough data
        if len(perf_data) > 10 and 'PnL' in perf_data.columns:
            returns = perf_data['PnL'].pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(252 * 6.5 * 60)
                sharpe_value = f"{sharpe:.2f}"

        # Calculate max drawdown
        if 'PnL' in perf_data.columns:
            cummax = perf_data['PnL'].cummax()
            drawdown = (cummax - perf_data['PnL']) / cummax
            max_dd = drawdown.max() * 100
            drawdown_value = f"{max_dd:.2f}%"

        # Count trades (assuming we track this somewhere)
        if 'Position' in perf_data.columns:
            position_changes = perf_data['Position'].diff().abs()
            trades = position_changes[position_changes > 0].count()
            trades_value = f"{trades}"

        # PnL Chart
        pnl_fig = go.Figure()
        if 'PnL' in perf_data.columns:
            pnl_fig.add_trace(go.Scatter(
                x=perf_data.index,
                y=perf_data['PnL'],
                mode='lines',
                name='P&L'
            ))
            pnl_fig.update_layout(
                title='Portfolio Value Over Time',
                xaxis_title='Simulation Step',
                yaxis_title='Portfolio Value ($)',
                template='plotly_dark'
            )

        # Position Chart
        position_fig = go.Figure()
        if 'Position' in perf_data.columns:
            position_fig.add_trace(go.Scatter(
                x=perf_data.index,
                y=perf_data['Position'],
                mode='lines',
                name='Position'
            ))
            position_fig.update_layout(
                title='Position Size Over Time',
                xaxis_title='Simulation Step',
                yaxis_title='Position Size',
                template='plotly_dark'
            )

    # Price Chart and Order Book if market data is available
    if market_data is not None and not market_data.empty:
        # Price Chart
        if 'Price' in market_data.columns:
            price_fig = go.Figure()
            price_fig.add_trace(go.Scatter(
                x=market_data.index,
                y=market_data['Price'],
                mode='lines',
                name='Price'
            ))
            price_fig.update_layout(
                title='Asset Price Over Time',
                xaxis_title='Simulation Step',
                yaxis_title='Price ($)',
                template='plotly_dark'
            )

        # Order Book Heatmap
        # This is simplified - in reality, you'd need to extract bid/ask levels from market data
        orderbook_fig = go.Figure()

        # If we have order book data (this is a placeholder)
        if 'BidPrice1' in market_data.columns and 'AskPrice1' in market_data.columns:
            # Create a sample heatmap from most recent order book state
            latest_idx = market_data.index[-1]

            # Extract bid levels
            bid_prices = []
            bid_sizes = []
            for i in range(1, 6):  # Assuming 5 levels
                if f'BidPrice{i}' in market_data.columns and f'BidSize{i}' in market_data.columns:
                    bid_prices.append(market_data[f'BidPrice{i}'].iloc[-1])
                    bid_sizes.append(market_data[f'BidSize{i}'].iloc[-1])

            # Extract ask levels
            ask_prices = []
            ask_sizes = []
            for i in range(1, 6):  # Assuming 5 levels
                if f'AskPrice{i}' in market_data.columns and f'AskSize{i}' in market_data.columns:
                    ask_prices.append(market_data[f'AskPrice{i}'].iloc[-1])
                    ask_sizes.append(market_data[f'AskSize{i}'].iloc[-1])

            # Combine and normalize for heatmap
            all_prices = bid_prices + ask_prices
            all_sizes = bid_sizes + ask_sizes

            # Create labels
            labels = [f"Bid {i+1}" for i in range(len(bid_prices))]
            labels.extend([f"Ask {i+1}" for i in range(len(ask_prices))])

            # Create heatmap
            orderbook_fig = go.Figure(data=go.Heatmap(
                z=[all_sizes],  # Single row
                x=labels,
                y=['Order Book'],
                colorscale='Blues',
                showscale=True
            ))

            orderbook_fig.update_layout(
                title='Order Book Depth',
                xaxis_title='Level',
                template='plotly_dark'
            )
        else:
            # Placeholder order book visualization
            orderbook_fig = go.Figure()
            orderbook_fig.add_trace(go.Bar(
                x=['Bid 5', 'Bid 4', 'Bid 3', 'Bid 2', 'Bid 1', 'Ask 1', 'Ask 2', 'Ask 3', 'Ask 4', 'Ask 5'],
                y=[100, 150, 200, 300, 500, 450, 250, 180, 120, 90],
                marker_color=['rgba(0, 0, 255, 0.5)'] * 5 + ['rgba(255, 0, 0, 0.5)'] * 5
            ))
            orderbook_fig.update_layout(
                title='Order Book Depth (Sample)',
                xaxis_title='Level',
                yaxis_title='Size',
                template='plotly_dark'
            )

    return pnl_value, sharpe_value, drawdown_value, trades_value, pnl_fig, position_fig, orderbook_fig, price_fig

if __name__ == "__main__":
    app.run(debug=True)
