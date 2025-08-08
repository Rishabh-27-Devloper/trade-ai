"""
Trading simulator for Trading AI System
Executes dummy trades and calculates P&L with realistic slippage and fees
"""
import time
import logging
import threading
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

from app.db import get_db_session, SimulatedTrade, log_system_event
from app.features import FeatureGenerator

logger = logging.getLogger(__name__)

class TradingSimulator:
    """Simulates trading operations with realistic market conditions"""

    def __init__(self, db_manager, config):
        self.db_manager = db_manager
        self.config = config
        self.feature_generator = FeatureGenerator(db_manager, config)

        self.running = False
        self.trainer = None  # Will be set by main app

        # Portfolio state
        self.initial_balance = 1000.0  # Starting with $10k
        self.current_balance = self.initial_balance
        self.open_positions = []
        self.last_trade_time = 0

        # Trading parameters
        self.position_size_usd = config.TRADE_AMOUNT_USDT
        self.max_positions = config.MAX_POSITIONS
        self.cooldown_seconds = config.COOLDOWN_SECONDS

        # Risk management
        self.stop_loss_percent = 0.005  # 0.5% stop loss
        self.take_profit_percent = 0.01  # 1% take profit (2:1 R:R)
        self.max_holding_hours = 3  # Max holding period

        # Fees and slippage (in basis points)
        self.slippage_bps = config.SLIPPAGE_BPS
        self.maker_fee_bps = config.MAKER_FEE_BPS
        self.taker_fee_bps = config.TAKER_FEE_BPS

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = self.initial_balance

    def set_trainer(self, trainer):
        """Set reference to model trainer"""
        self.trainer = trainer

    def run_continuously(self):
        """Run trading simulation continuously"""
        logger.info("📈 Starting continuous trading simulation...")
        self.running = True

        while self.running:
            try:
                # Check for trading opportunities
                self._check_trading_signals()

                # Manage open positions
                self._manage_open_positions()

                # Update performance metrics
                self._update_performance_metrics()

                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"❌ Error in simulation loop: {e}")
                time.sleep(60)

    def stop(self):
        """Stop trading simulation"""
        logger.info("🛑 Stopping trading simulation...")
        self.running = False

        # Close all open positions
        self._close_all_positions("system_stop")

    def _check_trading_signals(self):
        """Check for new trading signals from the model"""
        try:
            # Check cooldown period
            current_time = time.time()
            if current_time - self.last_trade_time < self.cooldown_seconds:
                return

            # Check if we can open new positions
            if len(self.open_positions) >= self.max_positions:
                return

            # Get latest market data and features
            latest_features = self.feature_generator.get_latest_features(1)

            if latest_features.empty:
                logger.debug("📊 No features available for signal generation")
                return

            # Get model prediction
            if not self.trainer or not self.trainer.current_model:
                logger.debug("🧠 No trained model available")
                return

            # Convert latest features to dictionary
            feature_dict = latest_features.iloc[-1].to_dict()

            # Add current price estimate (using close price from features calculation)
            current_price = self._get_current_price()
            if not current_price:
                logger.debug("💰 No current price available")
                return

            feature_dict['close_price'] = current_price

            # Get model prediction
            prediction = self.trainer.get_model_prediction(feature_dict)

            if not prediction:
                logger.debug("🔮 No prediction available")
                return

            # Check signal strength
            signal_strength = prediction.get('signal_strength', abs(prediction.get('probability', 0.5) - 0.5) * 2)

            # Only trade on strong signals
            min_signal_strength = 0.6
            if signal_strength < min_signal_strength:
                logger.debug(f"📊 Signal too weak: {signal_strength:.3f} < {min_signal_strength}")
                return

            # Determine trade direction
            trade_direction = 'long' if prediction['prediction'] == 1 else 'short'

            # Execute trade
            self._execute_trade(
                direction=trade_direction,
                signal=prediction['probability'],
                signal_strength=signal_strength,
                entry_price=current_price
            )

        except Exception as e:
            logger.error(f"❌ Error checking trading signals: {e}")

    def _execute_trade(self, direction: str, signal: float, signal_strength: float, entry_price: float):
        """Execute a simulated trade"""
        try:
            logger.info(f"⚡ Executing {direction} trade at ${entry_price:.2f} (signal: {signal:.3f}, strength: {signal_strength:.3f})")

            # Calculate position size based on risk
            position_size = self._calculate_position_size(entry_price)

            if position_size <= 0:
                logger.warning("⚠️ Position size too small, skipping trade")
                return

            # Apply slippage
            slipped_price = self._apply_slippage(entry_price, direction, 'entry')

            # Calculate fees
            entry_fee = self._calculate_fee(position_size * slipped_price, 'taker')

            # Calculate stop loss and take profit levels
            if direction == 'long':
                stop_loss_price = slipped_price * (1 - self.stop_loss_percent)
                take_profit_price = slipped_price * (1 + self.take_profit_percent)
            else:
                stop_loss_price = slipped_price * (1 + self.stop_loss_percent)
                take_profit_price = slipped_price * (1 - self.take_profit_percent)

            # Create trade record
            trade = SimulatedTrade(
                symbol=self.config.TRADING_PAIR,
                entry_timestamp=datetime.now(timezone.utc),
                entry_price=slipped_price,
                entry_signal=signal,
                position_size=position_size,
                side=direction,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                entry_fee=entry_fee,
                is_closed=False
            )

            # Save to database
            session = get_db_session()
            session.add(trade)
            session.commit()
            trade_id = trade.id
            session.close()

            # Add to open positions
            self.open_positions.append({
                'id': trade_id,
                'direction': direction,
                'entry_price': slipped_price,
                'position_size': position_size,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'entry_time': time.time(),
                'entry_fee': entry_fee,
                'max_adverse': 0.0,
                'max_favorable': 0.0
            })

            self.last_trade_time = time.time()
            self.total_trades += 1

            logger.info(f"✅ Trade executed: {direction} {position_size:.4f} {self.config.TRADING_PAIR} at ${slipped_price:.2f}")
            log_system_event('INFO', 'simulator', f'{direction} trade executed at ${slipped_price:.2f}')

        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")

    def _manage_open_positions(self):
        """Manage open positions - check for exits"""
        if not self.open_positions:
            return

        current_price = self._get_current_price()
        if not current_price:
            return

        current_time = time.time()
        positions_to_close = []

        for position in self.open_positions:
            try:
                # Update max adverse/favorable excursion
                if position['direction'] == 'long':
                    pnl_percent = (current_price - position['entry_price']) / position['entry_price']
                else:
                    pnl_percent = (position['entry_price'] - current_price) / position['entry_price']

                position['max_favorable'] = max(position['max_favorable'], pnl_percent)
                position['max_adverse'] = min(position['max_adverse'], pnl_percent)

                # Check exit conditions
                exit_reason = None

                # Stop loss check
                if position['direction'] == 'long' and current_price <= position['stop_loss']:
                    exit_reason = 'stop_loss'
                elif position['direction'] == 'short' and current_price >= position['stop_loss']:
                    exit_reason = 'stop_loss'

                # Take profit check
                elif position['direction'] == 'long' and current_price >= position['take_profit']:
                    exit_reason = 'take_profit'
                elif position['direction'] == 'short' and current_price <= position['take_profit']:
                    exit_reason = 'take_profit'

                # Time-based exit
                elif current_time - position['entry_time'] > self.max_holding_hours * 3600:
                    exit_reason = 'time_exit'

                if exit_reason:
                    positions_to_close.append((position, exit_reason, current_price))

            except Exception as e:
                logger.error(f"❌ Error managing position {position['id']}: {e}")

        # Close positions that need to be closed
        for position, reason, exit_price in positions_to_close:
            self._close_position(position, reason, exit_price)

    def _close_position(self, position: Dict, reason: str, exit_price: float):
        """Close an open position"""
        try:
            # Apply slippage to exit price
            slipped_exit_price = self._apply_slippage(exit_price, position['direction'], 'exit')

            # Calculate fees
            exit_fee = self._calculate_fee(position['position_size'] * slipped_exit_price, 'taker')

            # Calculate P&L
            if position['direction'] == 'long':
                gross_pnl = position['position_size'] * (slipped_exit_price - position['entry_price'])
            else:
                gross_pnl = position['position_size'] * (position['entry_price'] - slipped_exit_price)

            net_pnl = gross_pnl - position['entry_fee'] - exit_fee
            return_percent = (slipped_exit_price - position['entry_price']) / position['entry_price']
            if position['direction'] == 'short':
                return_percent *= -1

            holding_period_minutes = int((time.time() - position['entry_time']) / 60)

            # Update database record
            session = get_db_session()
            trade = session.query(SimulatedTrade).filter(SimulatedTrade.id == position['id']).first()

            if trade:
                trade.exit_timestamp = datetime.now(timezone.utc)
                trade.exit_price = slipped_exit_price
                trade.exit_reason = reason
                trade.pnl_gross = gross_pnl
                trade.pnl_net = net_pnl
                trade.return_percent = return_percent * 100  # Convert to percentage
                trade.holding_period_minutes = holding_period_minutes
                trade.max_adverse_excursion = position['max_adverse'] * 100
                trade.max_favorable_excursion = position['max_favorable'] * 100
                trade.exit_fee = exit_fee
                trade.slippage_cost = abs(exit_price - slipped_exit_price) * position['position_size']
                trade.is_closed = True

                session.commit()
            session.close()

            # Update performance tracking
            self.total_pnl += net_pnl
            self.current_balance += net_pnl

            if net_pnl > 0:
                self.winning_trades += 1

            # Remove from open positions
            self.open_positions = [p for p in self.open_positions if p['id'] != position['id']]

            logger.info(f"🏁 Position closed: {reason} at ${slipped_exit_price:.2f}, P&L: ${net_pnl:.2f} ({return_percent*100:.2f}%)")
            log_system_event('INFO', 'simulator', f'Position closed: {reason}, P&L: ${net_pnl:.2f}')

        except Exception as e:
            logger.error(f"❌ Error closing position: {e}")

    def _close_all_positions(self, reason: str):
        """Close all open positions"""
        current_price = self._get_current_price()
        if not current_price:
            return

        positions_to_close = self.open_positions.copy()
        for position in positions_to_close:
            self._close_position(position, reason, current_price)

    def _get_current_price(self) -> Optional[float]:
        """Get current market price"""
        try:
            # Get latest market data
            session = get_db_session()
            from app.db import MarketData

            latest_data = session.query(MarketData).filter(
                MarketData.symbol == self.config.TRADING_PAIR
            ).order_by(MarketData.timestamp.desc()).first()

            session.close()

            if latest_data:
                return float(latest_data.close_price)
            else:
                return None

        except Exception as e:
            logger.error(f"❌ Error getting current price: {e}")
            return None

    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on risk management"""
        # Calculate position size based on fixed USD amount
        position_size = self.position_size_usd / price

        # Apply risk management - don't risk more than available balance
        max_position_value = self.current_balance * 0.1  # Max 10% of balance per trade
        max_position_size = max_position_value / price

        return min(position_size, max_position_size)

    def _apply_slippage(self, price: float, direction: str, order_type: str) -> float:
        """Apply realistic slippage to price"""
        slippage_percent = self.slippage_bps / 10000.0  # Convert basis points to percentage

        if (direction == 'long' and order_type == 'entry') or (direction == 'short' and order_type == 'exit'):
            # Buying - price moves against us
            return price * (1 + slippage_percent)
        else:
            # Selling - price moves against us
            return price * (1 - slippage_percent)

    def _calculate_fee(self, trade_value: float, fee_type: str) -> float:
        """Calculate trading fees"""
        if fee_type == 'maker':
            fee_bps = self.maker_fee_bps
        else:
            fee_bps = self.taker_fee_bps

        return trade_value * (fee_bps / 10000.0)

    def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        try:
            # Update peak balance and drawdown
            if self.current_balance > self.peak_balance:
                self.peak_balance = self.current_balance

            current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

        except Exception as e:
            logger.error(f"❌ Error updating performance metrics: {e}")

    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        try:
            win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
            total_return = (self.current_balance - self.initial_balance) / self.initial_balance

            # Calculate current position values
            current_price = self._get_current_price()
            unrealized_pnl = 0.0

            if current_price and self.open_positions:
                for position in self.open_positions:
                    if position['direction'] == 'long':
                        pnl = position['position_size'] * (current_price - position['entry_price'])
                    else:
                        pnl = position['position_size'] * (position['entry_price'] - current_price)
                    unrealized_pnl += pnl

            return {
                'current_balance': round(self.current_balance, 2),
                'initial_balance': self.initial_balance,
                'total_pnl': round(self.total_pnl, 2),
                'unrealized_pnl': round(unrealized_pnl, 2),
                'total_return_percent': round(total_return * 100, 2),
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': round(win_rate * 100, 2),
                'max_drawdown_percent': round(self.max_drawdown * 100, 2),
                'open_positions': len(self.open_positions),
                'max_positions': self.max_positions,
                'position_size_usd': self.position_size_usd,
                'current_price': current_price,
                'running': self.running
            }

        except Exception as e:
            logger.error(f"❌ Error getting portfolio status: {e}")
            return {'error': str(e)}

    def get_recent_trades(self, limit: int = 20) -> list:
        """Get recent completed trades"""
        try:
            session = get_db_session()

            trades = session.query(SimulatedTrade).filter(
                SimulatedTrade.is_closed == True
            ).order_by(SimulatedTrade.exit_timestamp.desc()).limit(limit)

            results = []
            for trade in trades:
                results.append({
                    'id': trade.id,
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'entry_time': trade.entry_timestamp.isoformat(),
                    'exit_time': trade.exit_timestamp.isoformat() if trade.exit_timestamp else None,
                    'entry_price': round(trade.entry_price, 2),
                    'exit_price': round(trade.exit_price, 2) if trade.exit_price else None,
                    'position_size': round(trade.position_size, 4),
                    'pnl_net': round(trade.pnl_net, 2) if trade.pnl_net else None,
                    'return_percent': round(trade.return_percent, 2) if trade.return_percent else None,
                    'exit_reason': trade.exit_reason,
                    'holding_minutes': trade.holding_period_minutes
                })

            session.close()
            return results

        except Exception as e:
            logger.error(f"❌ Error getting recent trades: {e}")
            return []

    def manual_trade(self, direction: str, force: bool = False) -> bool:
        """Manually trigger a trade (for testing)"""
        try:
            if not force and len(self.open_positions) >= self.max_positions:
                logger.warning("⚠️ Maximum positions reached")
                return False

            current_price = self._get_current_price()
            if not current_price:
                logger.error("❌ No current price available")
                return False

            self._execute_trade(
                direction=direction,
                signal=0.8,  # Fake high confidence signal
                signal_strength=0.8,
                entry_price=current_price
            )

            return True

        except Exception as e:
            logger.error(f"❌ Manual trade failed: {e}")
            return False
