# web3app/compile_contract.py

import json
from solcx import compile_standard, install_solc

# Step 1: Install Solidity compiler version
install_solc("0.8.0")

# Step 2: Read Solidity contract
with open("web3app/contracts/ModelMetadata.sol", "r") as file:
    contract_source = file.read()

# Step 3: Compile the contract
compiled_sol = compile_standard({
    "language": "Solidity",
    "sources": {
        "ModelMetadata.sol": {
            "content": contract_source
        }
    },
    "settings": {
        "outputSelection": {
            "*": {
                "*": ["abi", "metadata", "evm.bytecode", "evm.sourceMap"]
            }
        }
    }
}, solc_version="0.8.0")

# Step 4: Extract ABI and Bytecode
contract_interface = compiled_sol['contracts']['ModelMetadata.sol']['ModelMetadata']
abi = contract_interface['abi']
bytecode = contract_interface['evm']['bytecode']['object']

# Step 5: Save ABI + Bytecode to build/ModelMetadata.json
with open("web3app/build/ModelMetadata.json", "w") as f:
    json.dump({
        "abi": abi,
        "bytecode": bytecode
    }, f, indent=4)

print(" Contract compiled successfully and ABI saved to build/ModelMetadata.json")
