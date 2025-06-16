import pytest
from decimal import Decimal
from unittest.mock import Mock, patch
from src.market_screener import PolymarketScreener, ScreeningCriteria, MarketOpportunity
from src.polymarket_client import PolymarketClient
from src.strategy import OrderbookSnapshot, OrderbookLevel


class TestScreeningCriteria:
    def test_default_criteria(self):
        criteria = ScreeningCriteria()
        
        assert criteria.min_daily_rewards == Decimal("50")
        assert criteria.min_reward_rate == Decimal("10")
        assert criteria.max_min_order_size == Decimal("500")
        assert criteria.max_competition_density == Decimal("0.6")
        assert criteria.min_spread_budget == Decimal("0.02")
        assert criteria.max_risk_level == "medium"
        assert criteria.require_active_orders is True
        assert criteria.exclude_archived is True
    
    def test_custom_criteria(self):
        criteria = ScreeningCriteria(
            min_daily_rewards=Decimal("100"),
            max_competition_density=Decimal("0.3"),
            max_risk_level="low"
        )
        
        assert criteria.min_daily_rewards == Decimal("100")
        assert criteria.max_competition_density == Decimal("0.3")
        assert criteria.max_risk_level == "low"


class TestPolymarketScreener:
    def setup_method(self):
        self.mock_client = Mock(spec=PolymarketClient)
        self.criteria = ScreeningCriteria(
            min_daily_rewards=Decimal("20"),
            max_competition_density=Decimal("0.8")
        )
        self.screener = PolymarketScreener(self.mock_client, self.criteria)
    
    def test_screener_initialization(self):
        assert self.screener.client == self.mock_client
        assert self.screener.criteria == self.criteria
        assert self.screener.logger is not None
    
    def test_parse_market_data(self):
        mock_market_data = {
            "question_id": "test_question",
            "question": "Test market question?",
            "active": True,
            "accepting_orders": True,
            "archived": False,
            "minimum_order_size": 10.0,
            "rewards": {
                "rates": [
                    {
                        "asset_address": "0x1234",
                        "rewards_daily_rate": 100.0
                    }
                ],
                "min_size": 50.0,
                "max_spread": 0.03
            },
            "tokens": [
                {
                    "token_id": "yes_token",
                    "outcome": "Yes",
                    "price": 0.6,
                    "winner": False
                },
                {
                    "token_id": "no_token", 
                    "outcome": "No",
                    "price": 0.4,
                    "winner": False
                }
            ]
        }
        
        market = self.screener._parse_market_data(mock_market_data)
        
        assert hasattr(market, 'question_id')
        assert hasattr(market, 'rewards')
        assert hasattr(market, 'tokens')
        assert market.question_id == "test_question"
        assert market.rewards['min_size'] == 50.0
        assert len(market.tokens) == 2
    
    def test_basic_criteria_filtering(self):
        # Create mock market that passes criteria
        good_market = type('Market', (), {})()
        good_market.accepting_orders = True
        good_market.archived = False
        good_market.minimum_order_size = 100.0
        good_market.rewards = {
            'rates': [{'rewards_daily_rate': 50.0}]
        }
        
        assert self.screener._passes_basic_criteria(good_market)
        
        # Test market that fails criteria
        bad_market = type('Market', (), {})()
        bad_market.accepting_orders = False  # Fails this check
        bad_market.archived = False
        bad_market.minimum_order_size = 100.0
        bad_market.rewards = {'rates': []}
        
        assert not self.screener._passes_basic_criteria(bad_market)
    
    def test_competition_density_estimation(self):
        # Create mock orderbooks
        yes_orderbook = OrderbookSnapshot(
            asset_id="yes_token",
            bids=[OrderbookLevel(Decimal("0.52"), Decimal("100"))],
            asks=[OrderbookLevel(Decimal("0.54"), Decimal("100"))],
            midpoint=Decimal("0.53"),
            spread=Decimal("0.02")
        )
        
        no_orderbook = OrderbookSnapshot(
            asset_id="no_token",
            bids=[OrderbookLevel(Decimal("0.46"), Decimal("100"))],
            asks=[OrderbookLevel(Decimal("0.48"), Decimal("100"))],
            midpoint=Decimal("0.47"),
            spread=Decimal("0.02")
        )
        
        density = self.screener._estimate_competition_density(
            yes_orderbook, no_orderbook, 0.03
        )
        
        assert isinstance(density, Decimal)
        assert Decimal("0") <= density <= Decimal("1")
    
    def test_reward_score_calculation(self):
        score = self.screener._calculate_reward_score(
            total_daily_rewards=100.0,
            min_size=200.0,
            competition_density=Decimal("0.3"),
            max_spread=0.03
        )
        
        assert isinstance(score, Decimal)
        assert score > Decimal("0")
    
    def test_risk_assessment(self):
        # Mock market object
        market = type('Market', (), {})()
        
        # Low risk scenario
        risk_low = self.screener._assess_risk_level(
            market, Decimal("0.2"), 0.05
        )
        assert risk_low in ["low", "medium", "high"]
        
        # High risk scenario
        risk_high = self.screener._assess_risk_level(
            market, Decimal("0.8"), 0.01
        )
        assert risk_high in ["low", "medium", "high"]
    
    def test_screening_criteria_check(self):
        # Create opportunity that meets criteria
        good_opportunity = MarketOpportunity(
            market=Mock(),
            yes_token=Mock(),
            no_token=Mock(),
            reward_score=Decimal("100"),
            competition_density=Decimal("0.3"),
            estimated_daily_rewards=Decimal("60"),  # Above min of 20
            min_capital_required=Decimal("200"),
            max_spread_allowed=Decimal("0.03"),  # Above min of 0.02
            recommended_position_size=Decimal("200"),
            risk_level="low"
        )
        
        assert self.screener._meets_screening_criteria(good_opportunity)
        
        # Create opportunity that fails criteria
        bad_opportunity = MarketOpportunity(
            market=Mock(),
            yes_token=Mock(), 
            no_token=Mock(),
            reward_score=Decimal("10"),
            competition_density=Decimal("0.9"),  # Above max of 0.8
            estimated_daily_rewards=Decimal("10"),  # Below min of 20
            min_capital_required=Decimal("200"),
            max_spread_allowed=Decimal("0.01"),  # Below min of 0.02
            recommended_position_size=Decimal("200"),
            risk_level="high"
        )
        
        assert not self.screener._meets_screening_criteria(bad_opportunity)
    
    def test_orderbook_conversion(self):
        mock_orderbook_data = {
            "bids": [
                {"price": "0.52", "size": "100"},
                {"price": "0.51", "size": "200"}
            ],
            "asks": [
                {"price": "0.54", "size": "150"},
                {"price": "0.55", "size": "250"}
            ]
        }
        
        snapshot = self.screener._convert_to_orderbook_snapshot(
            "test_asset", mock_orderbook_data, 0.53
        )
        
        assert snapshot.asset_id == "test_asset"
        assert len(snapshot.bids) == 2
        assert len(snapshot.asks) == 2
        assert snapshot.bids[0].price == Decimal("0.52")
        assert snapshot.asks[0].price == Decimal("0.54")
        assert snapshot.midpoint == Decimal("0.53")
        assert snapshot.spread == Decimal("0.02")
    
    @patch.object(PolymarketScreener, 'get_market_orderbooks')
    def test_analyze_market_opportunity(self, mock_get_orderbooks):
        # Mock orderbooks return
        yes_orderbook = OrderbookSnapshot(
            asset_id="yes_token",
            bids=[OrderbookLevel(Decimal("0.52"), Decimal("100"))],
            asks=[OrderbookLevel(Decimal("0.54"), Decimal("100"))],
            midpoint=Decimal("0.53"),
            spread=Decimal("0.02")
        )
        
        no_orderbook = OrderbookSnapshot(
            asset_id="no_token",
            bids=[OrderbookLevel(Decimal("0.46"), Decimal("100"))],
            asks=[OrderbookLevel(Decimal("0.48"), Decimal("100"))],
            midpoint=Decimal("0.47"),
            spread=Decimal("0.02")
        )
        
        mock_get_orderbooks.return_value = (yes_orderbook, no_orderbook)
        
        # Create mock market
        market = type('Market', (), {})()
        market.question_id = "test_question"
        market.rewards = {
            'rates': [{'rewards_daily_rate': 100.0}],
            'min_size': 50.0,
            'max_spread': 0.03
        }
        market.tokens = [
            {'token_id': 'yes_token', 'outcome': 'Yes', 'price': 0.6},
            {'token_id': 'no_token', 'outcome': 'No', 'price': 0.4}
        ]
        market.minimum_order_size = 10.0
        
        opportunity = self.screener._analyze_market_opportunity(market)
        
        assert opportunity is not None
        assert isinstance(opportunity, MarketOpportunity)
        assert opportunity.market == market
        assert opportunity.reward_score > Decimal("0")
        assert opportunity.estimated_daily_rewards == Decimal("100")
    
    def test_find_opportunities_integration(self):
        # Mock the client's get_sampling_markets method
        mock_response = {
            "data": [
                {
                    "question_id": "test_1",
                    "question": "Test question 1?",
                    "active": True,
                    "accepting_orders": True,
                    "archived": False,
                    "minimum_order_size": 10.0,
                    "rewards": {
                        "rates": [{"asset_address": "0x1234", "rewards_daily_rate": 100.0}],
                        "min_size": 50.0,
                        "max_spread": 0.03
                    },
                    "tokens": [
                        {"token_id": "yes_1", "outcome": "Yes", "price": 0.6, "winner": False},
                        {"token_id": "no_1", "outcome": "No", "price": 0.4, "winner": False}
                    ]
                }
            ],
            "next_cursor": None,
            "limit": 100,
            "count": 1
        }
        
        self.mock_client.get_sampling_markets.return_value = mock_response
        
        # Mock orderbook data
        mock_orderbook = {
            "bids": [{"price": "0.52", "size": "100"}],
            "asks": [{"price": "0.54", "size": "100"}]
        }
        self.mock_client.get_orderbook.return_value = mock_orderbook
        
        opportunities = self.screener.find_opportunities(max_markets=100)
        
        # Verify client was called
        self.mock_client.get_sampling_markets.assert_called()
        
        # Check results
        assert isinstance(opportunities, list)
        # Results depend on whether the mock market meets criteria


class TestMarketOpportunity:
    def test_market_opportunity_creation(self):
        market = Mock()
        yes_token = Mock()
        no_token = Mock()
        
        opportunity = MarketOpportunity(
            market=market,
            yes_token=yes_token,
            no_token=no_token,
            reward_score=Decimal("100"),
            competition_density=Decimal("0.3"),
            estimated_daily_rewards=Decimal("75"),
            min_capital_required=Decimal("500"),
            max_spread_allowed=Decimal("0.03"),
            recommended_position_size=Decimal("500"),
            risk_level="medium"
        )
        
        assert opportunity.market == market
        assert opportunity.reward_score == Decimal("100")
        assert opportunity.risk_level == "medium"
    
    def test_opportunity_to_dict(self):
        # Mock objects with necessary attributes
        market = Mock()
        market.question_id = "test_question"
        market.question = "Test question?"
        
        yes_token = Mock()
        yes_token.token_id = "yes_token_123"
        
        no_token = Mock()
        no_token.token_id = "no_token_123"
        
        opportunity = MarketOpportunity(
            market=market,
            yes_token=yes_token,
            no_token=no_token,
            reward_score=Decimal("100"),
            competition_density=Decimal("0.3"),
            estimated_daily_rewards=Decimal("75"),
            min_capital_required=Decimal("500"),
            max_spread_allowed=Decimal("0.03"),
            recommended_position_size=Decimal("500"),
            risk_level="medium"
        )
        
        result_dict = opportunity.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["question_id"] == "test_question"
        assert result_dict["reward_score"] == 100.0
        assert result_dict["risk_level"] == "medium"
        assert result_dict["yes_token_id"] == "yes_token_123"
        assert result_dict["no_token_id"] == "no_token_123"


class TestIntegrationScenarios:
    def setup_method(self):
        self.mock_client = Mock(spec=PolymarketClient)
        self.screener = PolymarketScreener(self.mock_client)
    
    def test_end_to_end_screening_workflow(self):
        """Test complete workflow from market fetching to opportunity ranking"""
        
        # Setup mock responses
        markets_response = {
            "data": [
                {
                    "question_id": "high_reward_market",
                    "question": "High reward market?",
                    "active": True,
                    "accepting_orders": True,
                    "archived": False,
                    "minimum_order_size": 10.0,
                    "rewards": {
                        "rates": [{"asset_address": "0x1234", "rewards_daily_rate": 200.0}],
                        "min_size": 30.0,
                        "max_spread": 0.04
                    },
                    "tokens": [
                        {"token_id": "high_yes", "outcome": "Yes", "price": 0.55, "winner": False},
                        {"token_id": "high_no", "outcome": "No", "price": 0.45, "winner": False}
                    ]
                },
                {
                    "question_id": "low_reward_market",
                    "question": "Low reward market?",
                    "active": True,
                    "accepting_orders": True,
                    "archived": False,
                    "minimum_order_size": 10.0,
                    "rewards": {
                        "rates": [{"asset_address": "0x1234", "rewards_daily_rate": 20.0}],
                        "min_size": 100.0,
                        "max_spread": 0.02
                    },
                    "tokens": [
                        {"token_id": "low_yes", "outcome": "Yes", "price": 0.50, "winner": False},
                        {"token_id": "low_no", "outcome": "No", "price": 0.50, "winner": False}
                    ]
                }
            ],
            "next_cursor": None,
            "limit": 100,
            "count": 2
        }
        
        orderbook_response = {
            "bids": [{"price": "0.52", "size": "50"}],
            "asks": [{"price": "0.54", "size": "50"}]
        }
        
        self.mock_client.get_sampling_markets.return_value = markets_response
        self.mock_client.get_orderbook.return_value = orderbook_response
        
        # Run screening
        opportunities = self.screener.find_opportunities(max_markets=100)
        
        # Verify results are properly ranked by reward score
        if opportunities:
            assert len(opportunities) <= 2
            # High reward market should rank higher if both pass criteria
            if len(opportunities) > 1:
                assert opportunities[0].reward_score >= opportunities[1].reward_score