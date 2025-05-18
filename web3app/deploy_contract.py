# web3app/deploy_contract.py

import json
import os
from web3 import Web3
from datetime import datetime

#  Config - Replace these with YOUR Ganache values
GANACHE_URL = "http://127.0.0.1:7545"
GANACHE_ADDRESS = "0xDd995B87FED357a5bc2A4049a92a646a7A6E3e66"  # Replace with your address
GANACHE_PRIVATE_KEY = "0x281d4cdd56379b1872ebb4ed0617934db7f0c0ecaa794c9b641b97ba57b7c2fd"  # Replace with your private key

#  Load ABI and Bytecode
with open("web3app/build/ModelMetadata.json", "r") as f:
    contract_data = json.load(f)

abi = contract_data["abi"]
bytecode = contract_data["bytecode"]

#  Connect to Ganache
w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
if not w3.is_connected():
    raise Exception(" Web3 not connected to Ganache")

#  Get nonce
nonce = w3.eth.get_transaction_count(GANACHE_ADDRESS)

#  Deploy contract
ModelMetadata = w3.eth.contract(abi=abi, bytecode=bytecode)
transaction = ModelMetadata.constructor().build_transaction({
    "from": GANACHE_ADDRESS,
    "nonce": nonce,
    "gas": 2000000,
    "gasPrice": w3.to_wei("50", "gwei")
})

#  Sign and send
signed_txn = w3.eth.account.sign_transaction(transaction, private_key=GANACHE_PRIVATE_KEY)
tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
print(f"⏳ Waiting for transaction {tx_hash.hex()} to be mined...")
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

#  Save contract address to file
contract_address = tx_receipt.contractAddress
with open("web3app/contract_address.txt", "w") as f:
    f.write(contract_address)

print(f" Contract deployed at: {contract_address}")
