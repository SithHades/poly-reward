import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Optional

from src.polymarket_client import PolymarketClient
from .config import MarketMakingConfig
from .risk_manager import RiskManager
from .order_manager import OrderManager
from .strategy import MarketMakingStrategy


class MarketMakingBot:
    """Main market making bot that orchestrates all components"""
    
    def __init__(self, config: MarketMakingConfig, client: Optional[PolymarketClient] = None):
        self.config = config
        self.client = client or PolymarketClient()
        self.logger = logging.getLogger(f"{__name__}.MarketMakingBot")
        
        # Initialize components
        self.risk_manager = RiskManager(config, self.client)
        self.order_manager = OrderManager(config, self.client)
        self.strategy = MarketMakingStrategy(config, self.client, self.risk_manager, self.order_manager)
        
        # Bot state
        self.is_running = False
        self.start_time = None
        self.last_health_check = datetime.now()
        self.health_check_interval = 300  # 5 minutes
        
        # Performance metrics
        self.cycles_completed = 0
        self.uptime_seconds = 0
        self.errors_encountered = 0
        
        # Track current task for signal handling
        self._current_task = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()
        
        # Cancel the current asyncio task if it exists
        if self._current_task and not self._current_task.done():
            self.logger.info("Cancelling current asyncio task...")
            self._current_task.cancel()
            
    async def _interruptible_sleep(self, duration: float):
        """Sleep for duration but check is_running flag every second for quick shutdown"""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < duration:
            if not self.is_running:
                self.logger.debug("Sleep interrupted due to shutdown")
                break
                
            # Sleep for 1 second at a time, or remaining duration if less than 1 second
            remaining = duration - (datetime.now() - start_time).total_seconds()
            sleep_time = min(1.0, remaining)
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
    async def start(self):
        """Start the market making bot"""
        
        if self.is_running:
            self.logger.warning("Bot is already running")
            return
            
        self.logger.info("Starting Market Making Bot")
        self.logger.info(f"Configuration: {self.config}")
        
        if self.config.dry_run:
            self.logger.warning("*** RUNNING IN DRY RUN MODE - NO REAL ORDERS WILL BE PLACED ***")
            
        self.is_running = True
        self.start_time = datetime.now()
        
        try:
            await self._main_loop()
        except Exception as e:
            self.logger.error(f"Fatal error in main loop: {e}")
            self.errors_encountered += 1
            raise
        finally:
            await self._cleanup()
            
    async def _main_loop(self):
        """Main bot loop"""
        
        self.logger.info("Entering main trading loop")
        
        while self.is_running:
            try:
                cycle_start = datetime.now()
                
                # Check if emergency stop is active
                if self.risk_manager.emergency_stop_active:
                    self.logger.error("Emergency stop active - shutting down bot")
                    self.stop()
                    break
                
                # Run strategy cycle
                await self.strategy.run_strategy_cycle()
                self.cycles_completed += 1
                
                # Periodic health check
                if (datetime.now() - self.last_health_check).total_seconds() > self.health_check_interval:
                    await self._health_check()
                    self.last_health_check = datetime.now()
                
                # Calculate cycle time and sleep
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                sleep_duration = max(0, self.config.order_refresh_interval - cycle_duration)
                
                if sleep_duration > 0:
                    self.logger.debug(f"Cycle completed in {cycle_duration:.2f}s, sleeping for {sleep_duration:.2f}s")
                    # Sleep in small increments to allow for quick shutdown
                    await self._interruptible_sleep(sleep_duration)
                else:
                    self.logger.warning(f"Cycle took {cycle_duration:.2f}s, longer than refresh interval!")
                    
            except Exception as e:
                self.logger.error(f"Error in main loop cycle: {e}")
                self.errors_encountered += 1
                
                # Brief pause before retrying
                await self._interruptible_sleep(5)
                
                # Stop if too many consecutive errors
                if self.errors_encountered > 10:
                    self.logger.error("Too many consecutive errors, stopping bot")
                    self.stop()
                    break
                    
        self.logger.info("Main loop ended")
        
    async def _health_check(self):
        """Perform periodic health checks"""
        
        self.logger.info("Performing health check")
        
        try:
            # Update uptime
            if self.start_time:
                self.uptime_seconds = (datetime.now() - self.start_time).total_seconds()
                
            # Get comprehensive metrics
            risk_summary = self.risk_manager.get_risk_summary()
            order_summary = self.order_manager.get_order_summary()
            strategy_metrics = self.strategy.get_strategy_metrics()
            
            # Log health status
            self.logger.info("=== HEALTH CHECK SUMMARY ===")
            self.logger.info(f"Uptime: {self.uptime_seconds/3600:.1f} hours")
            self.logger.info(f"Cycles completed: {self.cycles_completed}")
            self.logger.info(f"Errors encountered: {self.errors_encountered}")
            
            self.logger.info(f"Account - Balance: ${self.client.get_collateral_balance():.2f}")
            
            self.logger.info(f"Risk - Total exposure: ${risk_summary['total_exposure']:.2f}")
            self.logger.info(f"Risk - Unrealized PnL: ${risk_summary['total_unrealized_pnl']:.2f}")
            self.logger.info(f"Risk - Realized PnL: ${risk_summary['total_realized_pnl']:.2f}")
            self.logger.info(f"Risk - Active positions: {risk_summary['active_positions']}")
            self.logger.info(f"Risk - Emergency stop: {risk_summary['emergency_stop_active']}")
            
            self.logger.info(f"Orders - Active: {order_summary['active_orders']}")
            self.logger.info(f"Orders - Fill rate: {order_summary['fill_rate']:.2%}")
            self.logger.info(f"Orders - Placed today: {order_summary['orders_placed_today']}")
            self.logger.info(f"Orders - Filled today: {order_summary['orders_filled_today']}")
            
            self.logger.info(f"Strategy - Active markets: {strategy_metrics['active_markets']}")
            self.logger.info(f"Strategy - Opportunity rate: {strategy_metrics['opportunity_rate']:.2%}")
            self.logger.info("=== END HEALTH CHECK ===")
            
            # Reset error count on successful health check
            if self.errors_encountered > 0:
                self.logger.info(f"Resetting error count from {self.errors_encountered} to 0")
                self.errors_encountered = 0
                
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self.errors_encountered += 1
            
    def stop(self):
        """Stop the market making bot"""
        
        if not self.is_running:
            self.logger.warning("Bot is not currently running")
            return
            
        self.logger.info("Stopping Market Making Bot")
        self.is_running = False
        
    async def _cleanup(self):
        """Cleanup resources and orders on shutdown"""
        
        self.logger.info("Starting cleanup process")
        
        try:
            # Cancel all active orders
            self.order_manager.cleanup_all_orders()
            
            # Final metrics report
            await self._health_check()
            
            self.logger.info("Cleanup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            
    async def run_for_duration(self, duration_hours: float):
        """Run the bot for a specific duration then stop"""
        
        self.logger.info(f"Running bot for {duration_hours} hours")
        
        # Track the current task for signal handling
        self._current_task = asyncio.current_task()
        
        # Start bot in background task
        bot_task = asyncio.create_task(self.start())
        
        try:
            # Wait for specified duration or until bot stops
            duration_sleep_task = asyncio.create_task(asyncio.sleep(duration_hours * 3600))
            
            # Wait for either the duration to complete or the bot task to finish
            done, pending = await asyncio.wait(
                [bot_task, duration_sleep_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
            # If duration completed first, stop the bot
            if duration_sleep_task in done:
                self.stop()
                # Wait for bot to finish cleanup
                try:
                    await bot_task
                except Exception as e:
                    self.logger.error(f"Error while stopping bot: {e}")
                    
        except asyncio.CancelledError:
            # Handle cancellation (e.g., from signal handler)
            self.logger.info("Bot run cancelled")
            self.stop()
            try:
                await bot_task
            except (Exception, asyncio.CancelledError):
                pass
        finally:
            # Clear the current task reference
            self._current_task = None
            
        self.logger.info(f"Bot run completed after {duration_hours} hours")
        
    def get_status(self) -> dict:
        """Get current bot status and metrics"""
        
        return {
            "is_running": self.is_running,
            "start_time": self.start_time,
            "uptime_hours": self.uptime_seconds / 3600 if self.start_time else 0,
            "cycles_completed": self.cycles_completed,
            "errors_encountered": self.errors_encountered,
            "dry_run": self.config.dry_run,
            "emergency_stop": self.risk_manager.emergency_stop_active,
            "risk_summary": self.risk_manager.get_risk_summary(),
            "order_summary": self.order_manager.get_order_summary(),
            "strategy_metrics": self.strategy.get_strategy_metrics()
        }


async def main():
    """Example usage of the market making bot"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('market_making_bot.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create configuration
    config = MarketMakingConfig(
        crypto="ethereum",
        dry_run=True,  # Start in dry run mode for safety
        base_position_size=25.0,
        min_spread_threshold=0.015,  # 1.5% minimum spread
        target_profit_margin=0.008,  # 0.8% target profit margin
    )
    
    # Create and start bot
    bot = MarketMakingBot(config)
    
    try:
        # Run for 1 hour as a test
        await bot.run_for_duration(1.0)
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt")
    except Exception as e:
        logging.error(f"Bot error: {e}")
    finally:
        logging.info("Bot shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())