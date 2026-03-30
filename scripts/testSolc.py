import sys
import os
import solcx
import json
import pyevmasm
import re


def func_solc(Automata_contract):
    solcx.install_solc("0.4.24")
    solcx.set_solc_version("0.4.24")
    current_version = solcx.get_solc_version()
    print(f"Solidity compiler version: {current_version}")

    compiled_sol = solcx.compile_source(
        Automata_contract, output_values=["abi", "bin", "bin-runtime"]
    )

    contracts_bytecode = {}
    for contract_id, contract_interface in compiled_sol.items():
        abi = contract_interface["abi"]
        full_bytecode = contract_interface["bin"]
        runtime_bytecode = contract_interface["bin-runtime"]

        # print(f"Contract: {contract_id}")
        # print("ABI:", json.dumps(abi, indent=2))
        # print("full_bytecode:", full_bytecode)
        # print("runtime_bytecode:", runtime_bytecode)

        contracts_bytecode[contract_id] = (full_bytecode, runtime_bytecode)

    return contracts_bytecode


def bytecode_to_opcodes(_bytecode):
    instructions = list(pyevmasm.disassemble_all(_bytecode))
    opcodes = []
    for instr in instructions:
        if instr.operand is not None:
            opcodes.append(f"{instr.mnemonic}")
            opcodes.append(f"0x{instr.operand:x}")
        else:
            opcodes.append(instr.mnemonic)
    return opcodes


def main(sol_path=None, contract_name=None, output_path=None):
    """
    Compile a Solidity file and extract runtime opcodes.

    Args:
        sol_path: Path to .sol file. Defaults to VulnerableLogistics.sol in test dataset.
        contract_name: Contract name to extract (e.g. "VulnerableLogistics").
                       If None, uses the first contract found.
        output_path: Path to write opcodes. Defaults to bytecode2.txt next to this script.
    """
    if sol_path is None:
        sol_path = os.path.join(
            os.path.dirname(__file__), "..", "test_smartcontract_dataset",
            "dataset_for_train", "VulnerableLogistics.sol"
        )
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "bytecode2.txt")

    with open(sol_path, "r") as file:
        Automata_contract = file.read()

    contracts_bytecode = func_solc(Automata_contract)

    for contract_id, (full_bytecode, runtime_bytecode) in contracts_bytecode.items():
        full_opcode = bytecode_to_opcodes(bytes.fromhex(full_bytecode))
        runtime_opcode = bytecode_to_opcodes(bytes.fromhex(runtime_bytecode))

    # Select the target contract
    if contract_name:
        contract_id = f"<stdin>:{contract_name}"
    else:
        contract_id = list(contracts_bytecode.keys())[0]

    current_full_bytecode, current_runtime_bytecode = contracts_bytecode[contract_id]

    runtime_opcode_without_metadatahash = current_runtime_bytecode[:-88]  # strip metadata hash
    runtime_opcode = bytecode_to_opcodes(
        bytes.fromhex(runtime_opcode_without_metadatahash)
    )

    with open(output_path, "w") as f:
        for opcode in runtime_opcode:
            f.write(opcode + "\n")
    print(f"target bytecode has been written into {output_path}")


if __name__ == "__main__":
    main()
