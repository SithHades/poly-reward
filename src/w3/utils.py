import logging

from web3 import Web3
from web3.middleware import (
    ExtraDataToPOAMiddleware,
    SignAndSendRawMiddlewareBuilder
)
from web3.gas_strategies.time_based import fast_gas_price_strategy


logger = logging.getLogger(__name__)

erc20_balance_of = """[{"constant": true,"inputs": [{"name": "_owner","type": "address"}],"name": "balanceOf","outputs": [{"name": "balance","type": "uint256"}],"payable": false,"stateMutability": "view","type": "function"}]"""
erc1155_balance_of = """[{"inputs": [{"internalType": "address","name": "account","type": "address"},{"internalType": "uint256","name": "id","type": "uint256"}],"name": "balanceOf","outputs": [{"internalType": "uint256","name": "","type": "uint256"}],"stateMutability": "view","type": "function"}]"""

DECIMALS = 10**6


def setup_web3(rpc_url: str, private_key: str):
    w3 = Web3(Web3.HTTPProvider(rpc_url))

    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    w3.middleware_onion.add(SignAndSendRawMiddlewareBuilder.build(private_key))
    w3.eth.default_account = w3.eth.account.from_key(private_key).address

    w3.eth.set_gas_price_strategy(fast_gas_price_strategy)

    return w3


def balance_of_erc20(w3: Web3, token: str, address: str):
        erc20 = w3.eth.contract(token, abi=erc20_balance_of)
        bal = None
        try:
            bal = erc20.functions.balanceOf(address).call()
        except Exception as e:
            logger.error(f"Error ERC20 balanceOf: {e}")
            raise e

        return bal

def balance_of_erc1155(
        w3: Web3, erc1155_address: str, holder_address: str, token_id: int
    ):
        assert isinstance(erc1155_address, str)
        assert isinstance(holder_address, str)
        assert isinstance(token_id, int)

        erc1155 = w3.eth.contract(erc1155_address, abi=erc1155_balance_of)
        bal = None

        try:
            bal = erc1155.functions.balanceOf(holder_address, token_id).call()
        except Exception as e:
            logger.error(f"Error ERC1155 balanceOf: {e}")
            raise e

        return bal

def token_balance_of(w3: Web3, token: str, address: str, token_id=None):
        if token_id is None:
            bal = balance_of_erc20(w3, token, address)
        else:
            bal = balance_of_erc1155(w3,token, address, token_id)
        return float(bal / DECIMALS)
