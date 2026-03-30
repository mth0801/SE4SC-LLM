import z3
from z3 import Solver, BitVec, is_bv, simplify
from testSolc import func_solc, bytecode_to_opcodes
import re
import random
import time
from constants import STACK_MAX, SUCCESSOR_MAX, TEST_CASE_NUMBER_MAX, DEPTH_MAX, ICNT_MAX, SUBPATH_MAX, REWARD_MAX

MAX_TIME = 300  # Execution timeout per contract in seconds

# from graphviz import Digraph

name_to_value = {}

# def bfs_search_tree_node(node, target_index):
#     queue = deque([node])
#     while queue:
#         node = queue.popleft()
#         if target_index == node.bytecode_list_index:
#             return node
#         for child in node.children_node:
#             queue.append(child)
#     return TestTreeNode(-1, [], [], None)

class SymbolicVariableGenerator:
    def __init__(self):
        self.counter = 0

    def get_new_variable(self, name=None):
        if name is None:
            name = f"v{self.counter}"
            self.counter += 1
        var = BitVec(name, 256)

        name_to_value[var] = (
            int(name, 16)
            if name.startswith("0x") and "*" not in name and "&" not in name
            else name  # ???
        )
        return var


def convert_to_symbolic_bytecode(bytecode):
    symbolic_bytecode = []
    generator = SymbolicVariableGenerator()

    i = 0
    while i < len(bytecode):
        opcode = bytecode[i]
        if opcode.lower().startswith("push"):
            if opcode.lower() == "push0":
                symbolic_bytecode.append(opcode)
                i += 1
            else:
                value = generator.get_new_variable(bytecode[i + 1])
                symbolic_bytecode.append(opcode)
                symbolic_bytecode.append(value)
                i += 2
        else:
            symbolic_bytecode.append(opcode)
            i += 1
    return symbolic_bytecode

def remove_last_if_duplicate(lst):
    index = 0
    length = len(lst)
    while (index < length - 1):
        if (lst[index][1] == lst[length - 1][1]):
            if lst[index][0] <= lst[length - 1][0]:
                del lst[-1]
                return lst
            else:
                del lst[index]
                return lst
        index += 1
    return lst

class OpcodeHandlers:
    def __init__(self, executor, generator):
        self.executor = executor
        self.generator = generator

    def stop(self):
        self.executor.passed_program_paths[-1][1] = self.executor.bytecode_list_index
        ###
        temp_range = self.executor.passed_program_paths[-1]
        temp_range = f"{temp_range}"
        if temp_range in self.executor.passed_program_paths_to_passed_number:
            self.executor.passed_program_paths_to_passed_number[temp_range] += 1
        else:
            self.executor.passed_program_paths_to_passed_number[temp_range] = 1
        ### 
        self.executor.passed_program_paths = remove_last_if_duplicate(self.executor.passed_program_paths)

        # print("***************************************************")
        # print(self.executor.temp_node.branch_new_instruction)
        # print("***************************************************")
        self.executor.bytecode_list_index = len(self.executor.symbolic_bytecode)
        self.executor.test_case_num += 1
        self.executor.update_subpath_k4(self.executor.temp_node)

    def add(self):
        a = self.executor.stack.pop()
        b = self.executor.stack.pop()
        self.executor.stack.append(a + b)
        self.executor.bytecode_list_index += 1

    def mul(self):
        a = self.executor.stack.pop()
        b = self.executor.stack.pop()
        # self.executor.stack.append(a * b)
        self.executor.stack.append(self.generator.get_new_variable(f"{a}*{b}"))
        self.executor.bytecode_list_index += 1

    def sub(self):
        a = self.executor.stack.pop()
        b = self.executor.stack.pop()
        self.executor.stack.append(a - b)
        self.executor.bytecode_list_index += 1

    def div(self):
        a = self.executor.stack.pop()
        b = self.executor.stack.pop()
        if b == 0:
            self.executor.stack.append(0)
        else:
            self.executor.stack.append(a / b)
        self.executor.bytecode_list_index += 1

    def sdiv(self):
        a = self.executor.stack.pop()
        b = self.executor.stack.pop()
        if b == 0:
            self.executor.stack.append(0)
        else:
            self.executor.stack.append(a / b)
        self.executor.bytecode_list_index += 1

    def mod(self):
        a = self.executor.stack.pop()
        b = self.executor.stack.pop()
        if b == 0:
            self.executor.stack.append(0)
        else:
            self.executor.stack.append(a % b)
        self.executor.bytecode_list_index += 1

    def smod(self):
        a = self.executor.stack.pop()
        b = self.executor.stack.pop()
        if b == 0:
            self.executor.stack.append(0)
        else:
            self.executor.stack.append(a % b)
        self.executor.bytecode_list_index += 1

    def addmod(self):
        a = self.executor.stack.pop()
        b = self.executor.stack.pop()
        c = self.executor.stack.pop()
        self.executor.stack.append((a + b) % c)
        self.executor.bytecode_list_index += 1

    def mulmod(self):
        a = self.executor.stack.pop()
        b = self.executor.stack.pop()
        c = self.executor.stack.pop()
        self.executor.stack.append((a * b) % c)
        self.executor.bytecode_list_index += 1

    def exp(self):
        base = self.executor.stack.pop()
        exponent = self.executor.stack.pop()
        y = self.generator.get_new_variable(f"exp({base},{exponent})")
        self.executor.stack.append(y)
        self.executor.bytecode_list_index += 1

    def signextend(self):
        b = self.executor.stack.pop()
        x = self.executor.stack.pop()
        y = self.generator.get_new_variable(f"SIGNEXTEND({x},{b})")
        self.executor.stack.append(y)
        self.executor.bytecode_list_index += 1

    def lt(self):
        a = self.executor.stack.pop()
        b = self.executor.stack.pop()
        self.executor.stack.append(a < b)
        self.executor.bytecode_list_index += 1

    def gt(self):
        a = self.executor.stack.pop()
        b = self.executor.stack.pop()
        self.executor.stack.append(a > b)
        self.executor.bytecode_list_index += 1

    def slt(self):
        a = self.executor.stack.pop()
        b = self.executor.stack.pop()
        self.executor.stack.append(a < b)
        self.executor.bytecode_list_index += 1

    def sgt(self):
        a = self.executor.stack.pop()
        b = self.executor.stack.pop()
        self.executor.stack.append(a > b)
        self.executor.bytecode_list_index += 1

    def eq(self):
        a = self.executor.stack.pop()
        b = self.executor.stack.pop()
        self.executor.stack.append(a == b)
        self.executor.bytecode_list_index += 1

    def iszero(self):
        a = self.executor.stack.pop()
        self.executor.stack.append(a == False)
        self.executor.bytecode_list_index += 1

    def and_op(self):
        a = self.executor.stack.pop()
        b = self.executor.stack.pop()
        self.executor.stack.append(a & b)
        self.executor.bytecode_list_index += 1

    def or_op(self):
        a = self.executor.stack.pop()
        b = self.executor.stack.pop()
        self.executor.stack.append(a | b)
        self.executor.bytecode_list_index += 1

    def xor_op(self):
        a = self.executor.stack.pop()
        b = self.executor.stack.pop()
        self.executor.stack.append(a ^ b)
        self.executor.bytecode_list_index += 1

    def not_op(self):
        a = self.executor.stack.pop()
        self.executor.stack.append(~a)
        self.executor.bytecode_list_index += 1

    def byte_op(self):
        n = self.executor.stack.pop()
        x = self.executor.stack.pop()
        y = (x >> (248 - n * 8)) & 0xFF
        self.executor.stack.append(y)
        self.executor.bytecode_list_index += 1

    def shl(self):
        shift = self.executor.stack.pop()
        value = self.executor.stack.pop()
        self.executor.stack.append(value << shift)
        self.executor.bytecode_list_index += 1

    def shr(self):
        shift = self.executor.stack.pop()
        value = self.executor.stack.pop()
        self.executor.stack.append(value >> shift)
        self.executor.bytecode_list_index += 1

    def sar(self):
        shift = self.executor.stack.pop()
        value = self.executor.stack.pop()
        self.executor.stack.append(value >> shift)
        self.executor.bytecode_list_index += 1

    def sha3(self):
        offset = self.executor.stack.pop()
        length = self.executor.stack.pop()
        hash = self.generator.get_new_variable(
            f"keccak256(memory[{offset}:{offset}+{length}])"
        )
        self.executor.stack.append(hash)
        self.executor.bytecode_list_index += 1

    def address(self):
        address = self.generator.get_new_variable("address(this)")
        self.executor.stack.append(address)
        self.executor.bytecode_list_index += 1

    def balance(self):
        a = self.executor.stack.pop()
        balance = self.generator.get_new_variable(f"address({a}).balance")
        self.executor.stack.append(balance)
        self.executor.bytecode_list_index += 1

    def origin(self):
        origin = self.generator.get_new_variable("tx.origin")
        self.executor.stack.append(origin)
        self.executor.bytecode_list_index += 1

    def caller(self):
        caller = self.generator.get_new_variable("msg.caller")
        self.executor.stack.append(caller)
        self.executor.bytecode_list_index += 1

    def callvalue(self):
        callvalue = self.generator.get_new_variable("msg.value")
        self.executor.stack.append(callvalue)
        self.executor.bytecode_list_index += 1

    def calldataload(self):
        a = self.executor.stack.pop()
        callvalue = self.generator.get_new_variable(f"msg.data[{a}:{a}+32]")
        self.executor.stack.append(callvalue)
        self.executor.bytecode_list_index += 1

    def calldatasize(self):
        calldatasize = self.generator.get_new_variable("msg.data.size")
        self.executor.stack.append(calldatasize)
        self.executor.bytecode_list_index += 1

    def calldatacopy(self):  #
        dest = self.executor.stack.pop()
        offset = self.executor.stack.pop()
        length = self.executor.stack.pop()
        self.executor.bytecode_list_index += 1

    def codesize(self):
        size = self.generator.get_new_variable("address(this).code.size")
        self.executor.stack.append(size)
        self.executor.bytecode_list_index += 1

    def codecopy(self):
        dest = self.executor.stack.pop()
        offset = self.executor.stack.pop()
        length = self.executor.stack.pop()
        self.executor.bytecode_list_index += 1

    def gasprice(self):
        price = self.generator.get_new_variable("tx.gasprice")
        self.executor.stack.append(price)
        self.executor.bytecode_list_index += 1

    def extcodesize(self):
        address = self.executor.stack.pop()
        size = self.generator.get_new_variable(f"address({address}).code.size")
        self.executor.stack.append(size)
        self.executor.bytecode_list_index += 1

    def extcodecopy(self):
        self.executor.stack.pop()
        self.executor.stack.pop()
        self.executor.stack.pop()
        self.executor.stack.pop()
        self.executor.bytecode_list_index += 1

    def returndatasize(self):
        size = self.generator.get_new_variable("size_RETURNDATASIZE()")
        self.executor.stack.append(size)
        self.executor.bytecode_list_index += 1

    def returndatacopy(self):
        self.executor.stack.pop()
        self.executor.stack.pop()
        self.executor.stack.pop()
        self.executor.bytecode_list_index += 1

    def extcodehash(self):
        address = self.executor.stack.pop()
        hash_val = self.generator.get_new_variable(f"extcodehash_{address}")
        self.executor.stack.append(hash_val)
        self.executor.bytecode_list_index += 1

    def blockhash(self):
        block_number = self.executor.stack.pop()
        hash_val = self.generator.get_new_variable(f"blockhash_{block_number}")
        self.executor.stack.append(hash_val)
        self.executor.bytecode_list_index += 1

    def coinbase(self):
        coinbase = self.generator.get_new_variable("block.coinbase")
        self.executor.stack.append(coinbase)
        self.executor.bytecode_list_index += 1

    def timestamp(self):
        timestamp = self.generator.get_new_variable("block.timestamp")
        self.executor.stack.append(timestamp)
        self.executor.bytecode_list_index += 1

    def number(self):
        number = self.generator.get_new_variable("block.number")
        self.executor.stack.append(number)
        self.executor.bytecode_list_index += 1

    def difficulty(self):
        difficulty = self.generator.get_new_variable("block.difficulty")
        self.executor.stack.append(difficulty)
        self.executor.bytecode_list_index += 1

    def gaslimit(self):
        gaslimit = self.generator.get_new_variable("block.gaslimit")
        self.executor.stack.append(gaslimit)
        self.executor.bytecode_list_index += 1

    def chainid(self):
        chainid = self.generator.get_new_variable("chain_id")
        self.executor.stack.append(chainid)
        self.executor.bytecode_list_index += 1

    def selfbalance(self):
        balance = self.generator.get_new_variable("address(this).balance")
        self.executor.stack.append(balance)
        self.executor.bytecode_list_index += 1

    def basefee(self):
        basefee = self.generator.get_new_variable("base_fee")
        self.executor.stack.append(basefee)
        self.executor.bytecode_list_index += 1

    def blobhash(self):
        blob_index = self.executor.stack.pop()
        blobhash = self.generator.get_new_variable(
            f"tx.blob_versioned_hashes[{blob_index}]"
        )
        self.executor.stack.append(blobhash)
        self.executor.bytecode_list_index += 1

    def blobbasefee(self):
        blobbasefee = self.generator.get_new_variable("blob_base_fee")
        self.executor.stack.append(blobbasefee)
        self.executor.bytecode_list_index += 1

    def pop(self):
        self.executor.stack.pop()
        self.executor.bytecode_list_index += 1

    def mload(self):
        offset = self.executor.stack.pop()
        value = self.generator.get_new_variable(f"memory[{offset}:{offset}+32]")
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 1

    def mstore(self):
        address = self.executor.stack.pop()
        value = self.executor.stack.pop()
        self.executor.bytecode_list_index += 1

    def mstore8(self):
        address = self.executor.stack.pop()
        value = self.executor.stack.pop()
        self.executor.bytecode_list_index += 1

    def sload(self):
        key = self.executor.stack.pop()
        value = self.generator.get_new_variable(f"storage[{key}]")
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 1

    def sstore(self):
        key = self.executor.stack.pop()
        value = self.executor.stack.pop()
        self.executor.bytecode_list_index += 1

    def jump(self, jump_address_index):  # cunyi
        target = self.executor.stack.pop()
        if isinstance(jump_address_index, int):

            self.executor.passed_program_paths[-1][1] = self.executor.bytecode_list_index
            ###
            temp_range = self.executor.passed_program_paths[-1]
            temp_range = f"{temp_range}"
            if temp_range in self.executor.passed_program_paths_to_passed_number:
                self.executor.passed_program_paths_to_passed_number[temp_range] += 1
            else:
                self.executor.passed_program_paths_to_passed_number[temp_range] = 1
            ### 
            self.executor.passed_program_paths = remove_last_if_duplicate(self.executor.passed_program_paths)


            self.executor.execution_paths.append(
                (jump_address_index, self.executor.stack.copy())
            )

            if self.executor.way == "learch":
                next_node = TestTreeNode(jump_address_index, self.executor.stack.copy(), [], None)
            elif self.executor.way == "symflow":
                jumpSeq = self.executor.get_jumpSeq(jump_address_index)
                # print(f"jumpSeq:{jumpSeq}")
                next_node = TestTreeNode_symflow(jump_address_index, self.executor.stack.copy(), [], None, jumpSeq, "")  # JUMP is unconditional, no constraint
            
            next_node.parent_node = self.executor.temp_node
            next_node.successor_number = self.executor.find_successor_count(next_node.bytecode_list_index)
            next_node.test_case_number = self.executor.test_case_num
            next_node.branch_new_instruction, next_node.branch_new_instruction_pc_range = self.executor.count_branch_new_instruction(next_node.bytecode_list_index)
            if next_node.branch_new_instruction:
                next_node.branch_new_block = 1
            else:
                next_node.branch_new_block = 0
            
            if next_node.parent_node.is_occupied:
                next_node.path_new_instruction = 0
                next_node.path_new_block = 0
            else:
                next_node.path_new_instruction = next_node.parent_node.path_new_instruction + next_node.parent_node.branch_new_instruction
                next_node.path_new_block = next_node.parent_node.path_new_block + next_node.parent_node.branch_new_block
            
            next_node.depth = next_node.parent_node.depth + 1

            cpicnt_part = self.executor.count_cpicnt(next_node.parent_node.bytecode_list_index)
            next_node.cpicnt = next_node.parent_node.cpicnt + cpicnt_part
            
            passed_range = self.executor.count_execution_range(next_node.bytecode_list_index)
            passed_range = f"{passed_range}"
            if passed_range in self.executor.passed_program_paths_to_passed_number:
                next_node.icnt = self.executor.passed_program_paths_to_passed_number[passed_range]
            else:
                next_node.icnt = 0
            
            next_node.covNew = self.executor.count_covNew(next_node)

            next_node.subpath = self.executor.count_subpath_k4(next_node)
            
            self.executor.temp_node.add_child(next_node)
            self.executor.temp_nodes.append(next_node)

            self.executor.update_subpath_k4(self.executor.temp_node)

            ######
            self.executor.add_control_flow_edge(
                self.executor.bytecode_list_index, jump_address_index
            )


        else:
            raise ValueError("Invalid jump target")

    def jumpi(self, jumpi_address_index, jumpi_condition):  # cunyi
        self.executor.stack.pop()
        self.executor.stack.pop()

        if isinstance(jumpi_address_index, int):
            # print(f"jumpi_condition: {jumpi_condition}")
            # print(type(jumpi_condition))

            self.executor.passed_program_paths[-1][1] = self.executor.bytecode_list_index
            ###
            temp_range = self.executor.passed_program_paths[-1]
            temp_range = f"{temp_range}"
            if temp_range in self.executor.passed_program_paths_to_passed_number:
                self.executor.passed_program_paths_to_passed_number[temp_range] += 1
            else:
                self.executor.passed_program_paths_to_passed_number[temp_range] = 1
            ### 
            self.executor.passed_program_paths = remove_last_if_duplicate(self.executor.passed_program_paths)

            # print("***************************************************")
            # print(self.executor.temp_node.branch_new_instruction)
            # print("***************************************************")
            self.executor.solver.push()
            self.executor.solver.add(jumpi_condition != True)
            if self.executor.solver.check() == z3.sat:
                self.executor.execution_paths.append(
                    (jumpi_address_index, self.executor.stack.copy())
                )

                if self.executor.way == "learch":
                    false_node = TestTreeNode(jumpi_address_index, self.executor.stack.copy(), [], None)
                elif self.executor.way == "symflow":
                    jumpSeq = self.executor.get_jumpSeq(jumpi_address_index)
                    # Extract constraint: symbolic expression of the path condition
                    constraint_str = str(jumpi_condition) if jumpi_condition is not None else ""
                    # print(f"jumpSeq:{jumpSeq}")
                    false_node = TestTreeNode_symflow(jumpi_address_index, self.executor.stack.copy(), [], None, jumpSeq, constraint_str)

                false_node.parent_node = self.executor.temp_node
                false_node.successor_number = self.executor.find_successor_count(false_node.bytecode_list_index)
                false_node.test_case_number = self.executor.test_case_num
                false_node.branch_new_instruction, false_node.branch_new_instruction_pc_range = self.executor.count_branch_new_instruction(false_node.bytecode_list_index)
                if false_node.branch_new_instruction:
                    false_node.branch_new_block = 1
                else:
                    false_node.branch_new_block = 0

                if false_node.parent_node.is_occupied:
                    false_node.path_new_instruction = 0
                    false_node.path_new_block = 0
                else:
                    false_node.path_new_instruction = false_node.parent_node.path_new_instruction + false_node.parent_node.branch_new_instruction
                    false_node.path_new_block = false_node.parent_node.path_new_block + false_node.parent_node.branch_new_block
                
                false_node.depth = false_node.parent_node.depth + 1
                
                cpicnt_part = self.executor.count_cpicnt(false_node.parent_node.bytecode_list_index)
                false_node.cpicnt = false_node.parent_node.cpicnt + cpicnt_part

                passed_range = self.executor.count_execution_range(false_node.bytecode_list_index)
                passed_range = f"{passed_range}"
                if passed_range in self.executor.passed_program_paths_to_passed_number:
                    false_node.icnt = self.executor.passed_program_paths_to_passed_number[passed_range]
                else:
                    false_node.icnt = 0
                
                false_node.covNew = self.executor.count_covNew(false_node)

                false_node.subpath = self.executor.count_subpath_k4(false_node)

                self.executor.temp_node.add_child(false_node)
                self.executor.temp_nodes.append(false_node)

                self.executor.update_subpath_k4(self.executor.temp_node)
    
            self.executor.solver.pop()

            self.executor.add_control_flow_edge(
                self.executor.bytecode_list_index, jumpi_address_index
            )

            self.executor.solver.push()
            self.executor.solver.add(jumpi_condition == True)
            if self.executor.solver.check() == z3.sat:
                self.executor.execution_paths.append(
                    (
                        self.executor.bytecode_list_index + 1,
                        self.executor.stack.copy(),
                    )
                )

                if self.executor.way == "learch":
                    true_node = TestTreeNode(self.executor.bytecode_list_index + 1, self.executor.stack.copy(), [], None)
                elif self.executor.way == "symflow":
                    jumpSeq = self.executor.get_jumpSeq(self.executor.bytecode_list_index + 1)
                    # Extract constraint: symbolic expression of the path condition
                    constraint_str = str(jumpi_condition) if jumpi_condition is not None else ""
                    # print(f"jumpSeq:{jumpSeq}")
                    true_node = TestTreeNode_symflow(self.executor.bytecode_list_index + 1, self.executor.stack.copy(), [], None, jumpSeq, constraint_str)

                true_node.parent_node = self.executor.temp_node
                true_node.successor_number = self.executor.find_successor_count(true_node.bytecode_list_index)
                true_node.test_case_number = self.executor.test_case_num
                true_node.branch_new_instruction, true_node.branch_new_instruction_pc_range = self.executor.count_branch_new_instruction(true_node.bytecode_list_index)
                if true_node.branch_new_instruction:
                    true_node.branch_new_block = 1
                else:
                    true_node.branch_new_block = 0

                if true_node.parent_node.is_occupied:
                    true_node.path_new_instruction = 0
                    true_node.path_new_block = 0
                else:
                    true_node.path_new_instruction = true_node.parent_node.path_new_instruction + true_node.parent_node.branch_new_instruction
                    true_node.path_new_block = true_node.parent_node.path_new_block + true_node.parent_node.branch_new_block
                
                true_node.depth = true_node.parent_node.depth + 1

                cpicnt_part = self.executor.count_cpicnt(true_node.parent_node.bytecode_list_index)
                true_node.cpicnt = true_node.parent_node.cpicnt + cpicnt_part

                passed_range = self.executor.count_execution_range(true_node.bytecode_list_index)
                passed_range = f"{passed_range}"
                if passed_range in self.executor.passed_program_paths_to_passed_number:
                    true_node.icnt = self.executor.passed_program_paths_to_passed_number[passed_range]
                else:
                    true_node.icnt = 0

                true_node.covNew = self.executor.count_covNew(true_node)

                true_node.subpath = self.executor.count_subpath_k4(true_node)

                self.executor.temp_node.add_child(true_node)
                self.executor.temp_nodes.append(true_node)

                self.executor.update_subpath_k4(self.executor.temp_node)

            self.executor.solver.pop()

            self.executor.add_control_flow_edge(
                self.executor.bytecode_list_index, self.executor.bytecode_list_index + 1
            )

        else:
            raise ValueError("Invalid jumpi target")

    def pc(self):
        pc = self.generator.get_new_variable("PC")
        self.executor.stack.append(pc)
        self.executor.bytecode_list_index += 1

    def msize(self):
        msize = self.generator.get_new_variable("size_MSIZE()")
        self.executor.stack.append(msize)
        self.executor.bytecode_list_index += 1

    def gas(self):
        gas = self.generator.get_new_variable("gasRemaining")
        self.executor.stack.append(gas)
        self.executor.bytecode_list_index += 1

    def jumpdest(self):
        self.executor.bytecode_list_index += 1

    def tload(self):
        key = self.executor.stack.pop()
        value = self.generator.get_new_variable(f"transient[{key}]")
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 1

    def tstore(self):
        address = self.executor.stack.pop()
        value = self.executor.stack.pop()
        self.executor.bytecode_list_index += 1

    def mcopy(self):
        dest = self.executor.stack.pop()
        src = self.executor.stack.pop()
        length = self.executor.stack.pop()
        self.executor.bytecode_list_index += 1

    def push0(self):
        value = self.generator.get_new_variable("0")
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 1

    def push1(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push2(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push3(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push4(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push5(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push6(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push7(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push8(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push9(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push10(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push11(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push12(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push13(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push14(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push15(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push16(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push17(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push18(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push19(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push20(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push21(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push22(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push23(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push24(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push25(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push26(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push27(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push28(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push29(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push30(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push31(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def push32(self, value):
        self.executor.stack.append(value)
        self.executor.bytecode_list_index += 2

    def dup1(self):
        self.executor.stack.append(self.executor.stack[-1])
        self.executor.bytecode_list_index += 1

    def dup2(self):
        self.executor.stack.append(self.executor.stack[-2])
        self.executor.bytecode_list_index += 1

    def dup3(self):
        self.executor.stack.append(self.executor.stack[-3])
        self.executor.bytecode_list_index += 1

    def dup4(self):
        self.executor.stack.append(self.executor.stack[-4])
        self.executor.bytecode_list_index += 1

    def dup5(self):
        self.executor.stack.append(self.executor.stack[-5])
        self.executor.bytecode_list_index += 1

    def dup6(self):
        self.executor.stack.append(self.executor.stack[-6])
        self.executor.bytecode_list_index += 1

    def dup7(self):
        self.executor.stack.append(self.executor.stack[-7])
        self.executor.bytecode_list_index += 1

    def dup8(self):
        self.executor.stack.append(self.executor.stack[-8])
        self.executor.bytecode_list_index += 1

    def dup9(self):
        self.executor.stack.append(self.executor.stack[-9])
        self.executor.bytecode_list_index += 1

    def dup10(self):
        self.executor.stack.append(self.executor.stack[-10])
        self.executor.bytecode_list_index += 1

    def dup11(self):
        self.executor.stack.append(self.executor.stack[-11])
        self.executor.bytecode_list_index += 1

    def dup12(self):
        self.executor.stack.append(self.executor.stack[-12])
        self.executor.bytecode_list_index += 1

    def dup13(self):
        self.executor.stack.append(self.executor.stack[-13])
        self.executor.bytecode_list_index += 1

    def dup14(self):
        self.executor.stack.append(self.executor.stack[-14])
        self.executor.bytecode_list_index += 1

    def dup15(self):
        self.executor.stack.append(self.executor.stack[-15])
        self.executor.bytecode_list_index += 1

    def dup16(self):
        self.executor.stack.append(self.executor.stack[-16])
        self.executor.bytecode_list_index += 1

    def swap1(self):
        self.executor.stack[-1], self.executor.stack[-2] = (
            self.executor.stack[-2],
            self.executor.stack[-1],
        )
        self.executor.bytecode_list_index += 1

    def swap2(self):
        self.executor.stack[-1], self.executor.stack[-3] = (
            self.executor.stack[-3],
            self.executor.stack[-1],
        )
        self.executor.bytecode_list_index += 1

    def swap3(self):
        self.executor.stack[-1], self.executor.stack[-4] = (
            self.executor.stack[-4],
            self.executor.stack[-1],
        )
        self.executor.bytecode_list_index += 1

    def swap4(self):
        self.executor.stack[-1], self.executor.stack[-5] = (
            self.executor.stack[-5],
            self.executor.stack[-1],
        )
        self.executor.bytecode_list_index += 1

    def swap5(self):
        self.executor.stack[-1], self.executor.stack[-6] = (
            self.executor.stack[-6],
            self.executor.stack[-1],
        )
        self.executor.bytecode_list_index += 1

    def swap6(self):
        self.executor.stack[-1], self.executor.stack[-7] = (
            self.executor.stack[-7],
            self.executor.stack[-1],
        )
        self.executor.bytecode_list_index += 1

    def swap7(self):
        self.executor.stack[-1], self.executor.stack[-8] = (
            self.executor.stack[-8],
            self.executor.stack[-1],
        )
        self.executor.bytecode_list_index += 1

    def swap8(self):
        self.executor.stack[-1], self.executor.stack[-9] = (
            self.executor.stack[-9],
            self.executor.stack[-1],
        )
        self.executor.bytecode_list_index += 1

    def swap9(self):
        self.executor.stack[-1], self.executor.stack[-10] = (
            self.executor.stack[-10],
            self.executor.stack[-1],
        )
        self.executor.bytecode_list_index += 1

    def swap10(self):
        self.executor.stack[-1], self.executor.stack[-11] = (
            self.executor.stack[-11],
            self.executor.stack[-1],
        )
        self.executor.bytecode_list_index += 1

    def swap11(self):
        self.executor.stack[-1], self.executor.stack[-12] = (
            self.executor.stack[-12],
            self.executor.stack[-1],
        )
        self.executor.bytecode_list_index += 1

    def swap12(self):
        self.executor.stack[-1], self.executor.stack[-13] = (
            self.executor.stack[-13],
            self.executor.stack[-1],
        )
        self.executor.bytecode_list_index += 1

    def swap13(self):
        self.executor.stack[-1], self.executor.stack[-14] = (
            self.executor.stack[-14],
            self.executor.stack[-1],
        )
        self.executor.bytecode_list_index += 1

    def swap14(self):
        self.executor.stack[-1], self.executor.stack[-15] = (
            self.executor.stack[-15],
            self.executor.stack[-1],
        )
        self.executor.bytecode_list_index += 1

    def swap15(self):
        self.executor.stack[-1], self.executor.stack[-16] = (
            self.executor.stack[-16],
            self.executor.stack[-1],
        )
        self.executor.bytecode_list_index += 1

    def swap16(self):
        self.executor.stack[-1], self.executor.stack[-17] = (
            self.executor.stack[-17],
            self.executor.stack[-1],
        )
        self.executor.bytecode_list_index += 1

    def log0(self):
        data = self.executor.stack.pop()
        data = self.executor.stack.pop()
        self.executor.bytecode_list_index += 1

    def log1(self):
        data = self.executor.stack.pop()
        data = self.executor.stack.pop()
        data = self.executor.stack.pop()
        self.executor.bytecode_list_index += 1

    def log2(self):
        data = self.executor.stack.pop()
        data = self.executor.stack.pop()
        data = self.executor.stack.pop()
        data = self.executor.stack.pop()
        self.executor.bytecode_list_index += 1

    def log3(self):
        data = self.executor.stack.pop()
        data = self.executor.stack.pop()
        data = self.executor.stack.pop()
        data = self.executor.stack.pop()
        data = self.executor.stack.pop()
        self.executor.bytecode_list_index += 1

    def log4(self):
        data = self.executor.stack.pop()
        data = self.executor.stack.pop()
        data = self.executor.stack.pop()
        data = self.executor.stack.pop()
        data = self.executor.stack.pop()
        data = self.executor.stack.pop()
        self.executor.bytecode_list_index += 1

    def create(self):
        value = self.executor.stack.pop()
        offset = self.executor.stack.pop()
        length = self.executor.stack.pop()
        address = self.generator.get_new_variable(
            f"new_memory[{offset}:{offset}+{length}].value({value})"
        )
        self.executor.stack.append(address)
        self.executor.bytecode_list_index += 1

    def call(self):
        gas = self.executor.stack.pop()
        address = self.executor.stack.pop()
        value = self.executor.stack.pop()
        argsOffset = self.executor.stack.pop()
        argsLength = self.executor.stack.pop()
        retOffset = self.executor.stack.pop()
        retLength = self.executor.stack.pop()
        success = self.generator.get_new_variable("call_success")
        self.executor.stack.append(success)
        self.executor.bytecode_list_index += 1

    def callcode(self):
        gas = self.executor.stack.pop()
        address = self.executor.stack.pop()
        value = self.executor.stack.pop()
        argsOffset = self.executor.stack.pop()
        argsLength = self.executor.stack.pop()
        retOffset = self.executor.stack.pop()
        retLength = self.executor.stack.pop()
        success = self.generator.get_new_variable("callcode_success")
        self.executor.stack.append(success)
        self.executor.bytecode_list_index += 1

    def return_op(self):
        self.executor.stack.pop()
        self.executor.stack.pop()
        self.executor.passed_program_paths[-1][1] = self.executor.bytecode_list_index
        ###
        temp_range = self.executor.passed_program_paths[-1]
        temp_range = f"{temp_range}"
        if temp_range in self.executor.passed_program_paths_to_passed_number:
            self.executor.passed_program_paths_to_passed_number[temp_range] += 1
        else:
            self.executor.passed_program_paths_to_passed_number[temp_range] = 1
        ### 
        self.executor.passed_program_paths = remove_last_if_duplicate(self.executor.passed_program_paths)

        # print("***************************************************")
        # print(self.executor.temp_node.branch_new_instruction)
        # print("***************************************************")
        self.executor.bytecode_list_index = len(self.executor.symbolic_bytecode)
        self.executor.test_case_num += 1

        self.executor.update_subpath_k4(self.executor.temp_node)

    def delegatecall(self):
        gas = self.executor.stack.pop()
        address = self.executor.stack.pop()
        argsOffset = self.executor.stack.pop()
        argsLength = self.executor.stack.pop()
        retOffset = self.executor.stack.pop()
        retLength = self.executor.stack.pop()
        success = self.generator.get_new_variable("delegatecall_success")
        self.executor.stack.append(success)
        self.executor.bytecode_list_index += 1

    def create2(self):
        value = self.executor.stack.pop()
        offset = self.executor.stack.pop()
        length = self.executor.stack.pop()
        salt = self.executor.stack.pop()
        address = self.generator.get_new_variable(
            f"new_memory[{offset}:{offset}+{length}].value({value})"
        )
        self.executor.stack.append(address)
        self.executor.bytecode_list_index += 1

    def staticcall(self):
        gas = self.executor.stack.pop()
        address = self.executor.stack.pop()
        argsOffset = self.executor.stack.pop()
        argsLength = self.executor.stack.pop()
        retOffset = self.executor.stack.pop()
        retLength = self.executor.stack.pop()
        success = self.generator.get_new_variable("staticcall_success")
        self.executor.stack.append(success)
        self.executor.bytecode_list_index += 1

    def revert(self):
        self.executor.stack.pop()
        self.executor.stack.pop()

        self.executor.passed_program_paths[-1][1] = self.executor.bytecode_list_index
        ###
        temp_range = self.executor.passed_program_paths[-1]
        temp_range = f"{temp_range}"
        if temp_range in self.executor.passed_program_paths_to_passed_number:
            self.executor.passed_program_paths_to_passed_number[temp_range] += 1
        else:
            self.executor.passed_program_paths_to_passed_number[temp_range] = 1
        ### 
        self.executor.passed_program_paths = remove_last_if_duplicate(self.executor.passed_program_paths)

        # print("***************************************************")
        # print(self.executor.temp_node.branch_new_instruction)
        # print("***************************************************")
        self.executor.bytecode_list_index = len(self.executor.symbolic_bytecode)
        self.executor.test_case_num += 1

        self.executor.update_subpath_k4(self.executor.temp_node)

    def selfdestruct(self):
        address = self.executor.stack.pop()
    
        self.executor.passed_program_paths[-1][1] = self.executor.bytecode_list_index
        ###
        temp_range = self.executor.passed_program_paths[-1]
        temp_range = f"{temp_range}"
        if temp_range in self.executor.passed_program_paths_to_passed_number:
            self.executor.passed_program_paths_to_passed_number[temp_range] += 1
        else:
            self.executor.passed_program_paths_to_passed_number[temp_range] = 1
        ### 
        self.executor.passed_program_paths = remove_last_if_duplicate(self.executor.passed_program_paths)

        # print("***************************************************")
        # print(self.executor.temp_node.branch_new_instruction)
        # print("***************************************************")
        self.executor.bytecode_list_index = len(self.executor.symbolic_bytecode)
        self.executor.test_case_num += 1

        self.executor.update_subpath_k4(self.executor.temp_node)

    def invalid(self):
        self.executor.bytecode_list_index += 1
        # raise ValueError("Invalid opcode")

class BytecodeExecutor:
    def __init__(self, symbolic_bytecode, real_bytecode):
        self.symbolic_bytecode = symbolic_bytecode
        self.real_bytecode = real_bytecode
        self.stack = []
        self.bytecode_list_index = 0
        self.generator = SymbolicVariableGenerator()
        self.handlers = OpcodeHandlers(self, self.generator)
        self.index_mapping_pc, self.pc_mapping_index = (
            self.create_mapping()
        )
        self.solver = Solver()
        self.execution_paths = []
        self.paths = []
        self.smartcontract_functions_index_position = []
        self.smartcontract_functions_index_range = []
        self.stack_snapshots = {}
        self.control_flow_graph = {}
        self.visited_nodes_index_by_jumpi = {}
        self.exist_loop_node_by_jumpi = set()
        self.jump_structure_info = []
        self.opcodeindex_to_stack = {}
        self.all_jump_index_related_to_Call = set()
        self.contract_start_index = []
        self.the_first_contract_end_index = 0

    def create_mapping(self):
        index_mapping_pc = {}
        pc_mapping_index = {}
        pc = 0
        index = 0
        while index < len(self.symbolic_bytecode):
            # print(self.symbolic_bytecode[index])  #
            opcode = self.symbolic_bytecode[index].lower()
            index_mapping_pc[index] = pc
            pc_mapping_index[pc] = index
            pc += 1
            if opcode.startswith("push") and not opcode.startswith("push0"):

                push_bytes = int(opcode[4:])
                pc += push_bytes
                index += 1
            index += 1
        return index_mapping_pc, pc_mapping_index

    def get_pc_position(self, index):
        return self.index_mapping_pc.get(index, None)

    def get_index_position(self, pc):
        return self.pc_mapping_index.get(pc, None)

    def get_max_stop_return_index(self):

        contract_start_index = []
        index = 0
        while index < len(self.real_bytecode):
            if (
                self.real_bytecode[index].startswith("PUSH")
                and self.real_bytecode[index] != "PUSH0"
                and self.real_bytecode[index + 1] == "0x80"
                and self.real_bytecode[index + 2].startswith("PUSH")
                and self.real_bytecode[index + 2] != "PUSH0"
                and self.real_bytecode[index + 3] == "0x40"
                and self.real_bytecode[index + 4] == "MSTORE"
            ):
                contract_start_index.append(index)
            index += 1
        self.contract_start_index = contract_start_index
        # print(f"contract_start_index: {contract_start_index}")

        if len(contract_start_index) >= 2:
            # print("111")
            stop_return_indices = []
            index1 = 0
            while index1 < len(self.real_bytecode):
                if (
                    self.real_bytecode[index1] == "STOP"
                    or self.real_bytecode[index1] == "RETURN"
                ) and index1 < contract_start_index[1]:
                    stop_return_indices.append(index1)
                    # print(f"stop_return_indices: {stop_return_indices}")
                index1 += 1
            stop_return_indices.pop()
            dispatcher_boundary = max(stop_return_indices)
            self.the_first_contract_end_index = contract_start_index[1] - 2
        else:
            stop_return_indices = [
                index1
                for index1, opcode in enumerate(self.real_bytecode)
                if opcode.lower() == "stop" or opcode.lower() == "return"
            ]
            if len(stop_return_indices) == 0:
                return -1
            else:
                dispatcher_boundary = max(stop_return_indices)
            self.the_first_contract_end_index = len(self.real_bytecode) - 1

        # print(f"self.the_first_contract_end_index: {self.the_first_contract_end_index}")
        # print(f"stop_return_indices: {stop_return_indices}")
        # print(f"dispatcher_boundary: {dispatcher_boundary}")
        return dispatcher_boundary

    def record_stack_snapshot(self):
        self.stack_snapshots[self.bytecode_list_index] = len(self.stack)

    def record_stack(self):
        self.opcodeindex_to_stack[self.bytecode_list_index] = self.stack.copy()


    def add_control_flow_edge(self, source, target):
        if source in self.control_flow_graph:
            if target not in self.control_flow_graph[source]:
                self.control_flow_graph[source].append(target)
        else:
            self.control_flow_graph[source] = [target]

    def create_control_flow_graph(self):
        # print(f"control_flow_graph: {self.control_flow_graph}")
        return self.control_flow_graph
        # dot = Digraph()

        # for source, targets in self.control_flow_graph.items():
        #     for target in targets:
        #         dot.edge(f"PC {source}", f"PC {target}")
        # return dot

    def execute(self):
        dispatcher_boundary = self.get_max_stop_return_index()
        if dispatcher_boundary == -1:
            return False
        self.execution_paths = [(self.bytecode_list_index, self.stack.copy())]
        all_stacks = []
        self.smartcontract_functions_index_position.append(dispatcher_boundary + 1)
        while self.execution_paths:
            self.bytecode_list_index, self.stack = self.execution_paths.pop()  #
            while self.bytecode_list_index < len(self.real_bytecode):
                # print(f"index:{self.bytecode_list_index}")
                # print(f"stack:{self.stack}")
                opcode = self.symbolic_bytecode[self.bytecode_list_index]
                # print(opcode)
                handler_name = opcode.lower()
                handler = getattr(self.handlers, handler_name, None)
                self.record_stack_snapshot()
                self.record_stack()
                # print(
                #     f"self.opcodeindex_to_stack[self.bytecode_list_index]: {self.opcodeindex_to_stack[self.bytecode_list_index]}"
                # )

                if opcode.lower().startswith("push") and opcode.lower() != "push0":
                    value = self.symbolic_bytecode[self.bytecode_list_index + 1]
                    handler(value)
                elif opcode.lower() == "jump":
                    jump_address_symbol = self.stack[-1]


                    if "&" not in str(jump_address_symbol):
                        jump_address_pc = name_to_value[jump_address_symbol]
                    else:
                        parts = str(jump_address_symbol).split("&")

                        result = parts[1].strip()
                        jump_address_pc = name_to_value[BitVec(result, 256)]

                    jump_address_index = self.get_index_position(jump_address_pc)
                    print(name_to_value)
                    print(f"jump address pc: {self.stack[-1]}")
                    print(type(self.stack[-1]))
                    print(f"jump address pc: {jump_address_pc}")
                    print(f"jump address index: {jump_address_index}")


                    if jump_address_index <= dispatcher_boundary:

                        if self.bytecode_list_index + 1 > dispatcher_boundary + 2:
                            if (
                                self.bytecode_list_index + 1
                                not in self.smartcontract_functions_index_position
                            ):
                                self.smartcontract_functions_index_position.append(
                                    self.bytecode_list_index + 1
                                )
                                print(
                                    f"self.smartcontract_functions_index_position: {self.smartcontract_functions_index_position}"
                                )

                    handler(jump_address_index)

                    break
                    # exist another way
                elif opcode.lower() == "jumpi":
                    print(f"happen branch jumpi in index {self.bytecode_list_index}")
                    jumpi_address_symbol = self.stack[-1]
                    jumpi_condition = self.stack[-2]
                    jumpi_address_pc = name_to_value[jumpi_address_symbol]
                    jumpi_address_index = self.get_index_position(jumpi_address_pc)









                    if self.bytecode_list_index in self.visited_nodes_index_by_jumpi:

                        if (
                            self.visited_nodes_index_by_jumpi[self.bytecode_list_index]
                            <= 4
                        ):
                            self.visited_nodes_index_by_jumpi[
                                self.bytecode_list_index
                            ] += 1
                            handler(jumpi_address_index, jumpi_condition)
                        else:
                            print(
                                f"Exist Loop!!! Exist Loop!!! Exist Loop!!! in PC {self.bytecode_list_index}"
                            )

                            self.exist_loop_node_by_jumpi.add(self.bytecode_list_index)


                            # self.visited_nodes_index_by_jumpi[
                            #     self.bytecode_list_index
                            # ] = 1

                            break
                    else:
                        self.visited_nodes_index_by_jumpi.update(
                            {self.bytecode_list_index: 0}
                        )
                        handler(jumpi_address_index, jumpi_condition)

                    break  # ???
                elif (
                    opcode.lower() == "return"
                ):
                    handler_name = "return_op"
                    handler = getattr(self.handlers, handler_name, None)
                    handler()
                elif opcode.lower().startswith("and"):  # 。。。
                    handler_name = "and_op"
                    handler = getattr(self.handlers, handler_name, None)
                    handler()
                elif opcode.lower() == "or":  # 。。。
                    handler_name = "or_op"
                    handler = getattr(self.handlers, handler_name, None)
                    handler()
                elif opcode.lower() == "xor":  # 。。。
                    handler_name = "xor_op"
                    handler = getattr(self.handlers, handler_name, None)
                    handler()
                elif opcode.lower() == "not":  # 。。。
                    handler_name = "not_op"
                    handler = getattr(self.handlers, handler_name, None)
                    handler()
                elif opcode.lower() == "byte":  # 。。。
                    handler_name = "byte_op"
                    handler = getattr(self.handlers, handler_name, None)
                    handler()
                else:
                    handler()

                if handler is None:
                    raise ValueError(f"Unknown opcode: {opcode}")

            all_stacks.append(self.stack.copy())

        self.control_flow_graph = dict(sorted(self.control_flow_graph.items()))
        print(f"self.control_flow_graph: {self.control_flow_graph}")




        all_target_jumpdest_index = []
        for key in self.control_flow_graph.keys():
            all_target_jumpdest_index.extend(self.control_flow_graph[key])
        all_target_jumpdest_index = set(all_target_jumpdest_index)
        all_target_jumpdest_index = list(all_target_jumpdest_index)
        all_target_jumpdest_index = sorted(all_target_jumpdest_index)
        print(f"all_target_jumpdest_index: {all_target_jumpdest_index}")

        exist_internal_function_boundrys_index = set()
        index = dispatcher_boundary + 1
        while index <= self.the_first_contract_end_index:
            if (
                self.real_bytecode[index] == "JUMP"
                and self.real_bytecode[index - 2].startswith("PUSH")
                and self.real_bytecode[index - 2] != "PUSH0"
                and self.real_bytecode[index + 1] == "JUMPDEST"
            ):
                print(f"new turn!!!")
                repeated_jump_nodes = set()
                passed_paths_range_list = []
                possible_search_index = []
                possible_search_index.append(index)
                while possible_search_index:
                    current_index = possible_search_index.pop()
                    current_index_target_address_index_list = self.control_flow_graph[
                        current_index
                    ]
                    print(
                        f"current_index_target_address_index_list: {current_index_target_address_index_list}"
                    )
                    if current_index in repeated_jump_nodes:
                        continue
                    else:
                        repeated_jump_nodes.add(current_index)

                    for target_address_index in current_index_target_address_index_list:
                        keep_go_signal = True
                        next_jump_or_jumpi_index = min(
                            [
                                key
                                for key in self.control_flow_graph.keys()
                                if key > target_address_index
                            ]
                        )
                        for index1 in range(
                            target_address_index + 1, next_jump_or_jumpi_index
                        ):
                            if (
                                self.real_bytecode[index1] == "RETURN"
                                or self.real_bytecode[index1] == "STOP"
                                or self.real_bytecode[index1] == "REVERT"
                                or self.real_bytecode[index1] == "INVALID"
                                or self.real_bytecode[index1] == "SELFDESTRUCT"
                            ):
                                keep_go_signal = False
                                break
                        if keep_go_signal:
                            possible_search_index.append(next_jump_or_jumpi_index)
                            passed_paths_range_list.append(
                                [target_address_index, next_jump_or_jumpi_index]
                            )
                    print(f"passed_paths_range_list: {passed_paths_range_list}")
                print(f"repeated_jump_nodes: {repeated_jump_nodes}")
                for path in passed_paths_range_list:
                    if index + 1 in range(path[0], path[1] + 1):
                        print("111")
                        print(f"index+1: {index + 1}")
                        for key in self.control_flow_graph.keys():
                            if (
                                index + 1 in self.control_flow_graph[key]
                                and self.real_bytecode[key] == "JUMP"
                            ):
                                print("222")
                                self.all_jump_index_related_to_Call.add(index)
                                exist_internal_function_boundrys_index.add(key + 1)

            index += 1

        print(
            f"self.all_jump_index_related_to_Call: {self.all_jump_index_related_to_Call}"
        )
        print(
            f"exist_internal_function_boundrys_index: {exist_internal_function_boundrys_index}"
        )

        for index in list(exist_internal_function_boundrys_index):
            if index not in self.smartcontract_functions_index_position:
                self.smartcontract_functions_index_position.append(index)

        self.smartcontract_functions_index_position = sorted(
            self.smartcontract_functions_index_position
        )

        i = 0
        while i < len(self.smartcontract_functions_index_position) and i + 1 < len(
            self.smartcontract_functions_index_position
        ):
            self.smartcontract_functions_index_range.append(
                [
                    self.smartcontract_functions_index_position[i],
                    self.smartcontract_functions_index_position[i + 1] - 1,
                ]
            )
            i += 1
        print(self.smartcontract_functions_index_position)
        print("...")
        print(
            f"self.smartcontract_functions_index_range: {self.smartcontract_functions_index_range}"
        )


        if (
            len(self.smartcontract_functions_index_range) == 1
            and self.real_bytecode[-1] == "RETURN"
        ):
            self.smartcontract_functions_index_range[0][0] = (
                self.smartcontract_functions_index_range[0][0] + 2
            )
            print(
                f"final self.smartcontract_functions_index_range: {self.smartcontract_functions_index_range}"
            )

        return True


class TestTreeNode_symflow:
    def __init__(self, bytecode_list_index, stack, children_node, parent_node, jumpSeq, constraint=""):
        self.bytecode_list_index = bytecode_list_index
        self.stack_size = len(stack)
        self.children_node = children_node
        self.parent_node = parent_node
        self.successor_number = -1
        self.depth = 0
        self.cpicnt = 0
        self.covNew = 0
        self.executed = False

        self.jumpSeq = jumpSeq
        self.constraint = constraint

        # self.


        self.test_case_number = 0

        self.branch_new_instruction = 0
        self.branch_new_instruction_pc_range = [-2, -2]
        self.branch_new_block = 0
        
        self.is_occupied = False
        self.path_new_instruction = 0
        self.path_new_block = 0

        self.icnt = 0

        self.subpath = 0


        self.reward = 0

    def add_child(self, child_node):
        self.children_node.append(child_node)


class TestTreeNode:
    def __init__(self, bytecode_list_index, stack, children_node, parent_node):
        self.bytecode_list_index = bytecode_list_index
        self.stack_size = len(stack)
        self.children_node = children_node
        self.parent_node = parent_node
        self.successor_number = -1
        self.depth = 0
        self.cpicnt = 0
        self.covNew = 0
        self.executed = False


        self.test_case_number = 0

        self.branch_new_instruction = 0
        self.branch_new_instruction_pc_range = [-2, -2]
        self.branch_new_block = 0
        
        self.is_occupied = False
        self.path_new_instruction = 0
        self.path_new_block = 0

        self.icnt = 0

        self.subpath = 0


        self.reward = 0

    def add_child(self, child_node):
        self.children_node.append(child_node)


class SymExec(BytecodeExecutor):
    def __init__(self, symbolic_bytecode, real_bytecode, strategy, way, detector=None):
        super().__init__(symbolic_bytecode, real_bytecode)
        self.strategy = strategy
        self.way = way
        self.detector = detector  # Optional VulnerabilityDetector (inference phase only)
    
        self.test_case_num = 0
        self.passed_program_paths = []
        self.passed_program_paths_to_passed_number = {}
        self.storage_branch_new_instruction = 0

        if way == "learch":
            self.origin_node = TestTreeNode(0, [], [], None)
            self.origin_node.successor_number = self.find_successor_count(self.origin_node.bytecode_list_index)
            self.origin_node.branch_new_instruction, self.origin_node.branch_new_instruction_pc_range = self.count_branch_new_instruction(self.origin_node.bytecode_list_index)
            if self.origin_node.branch_new_instruction:
                self.origin_node.branch_new_block = 1
            else:
                self.origin_node.branch_new_block = 0
        elif way == "symflow":

            origin_jumpSeq = self.get_jumpSeq(0)

            self.origin_node = TestTreeNode_symflow(0, [], [], None, origin_jumpSeq, "")
            self.origin_node.successor_number = self.find_successor_count(self.origin_node.bytecode_list_index)
            self.origin_node.branch_new_instruction, self.origin_node.branch_new_instruction_pc_range = self.count_branch_new_instruction(self.origin_node.bytecode_list_index)
            if self.origin_node.branch_new_instruction:
                self.origin_node.branch_new_block = 1
            else:
                self.origin_node.branch_new_block = 0


        self.subpath_k4_to_number = {}


        self.dispatcher_boundary = -1

        self.all_jump_jumpi_number = self.count_smart_contract_jump_jumpi_number(self.real_bytecode)
        self.coverage = 0
        self.arrive_assigned_coverage_time = []
        self.assigned_coverage = 0.5
        self.select_state_accuracy_count = []
        self.select_state_accuracy = 0

        # Benchmark timing attributes
        self._predict_time_total = 0.0
        self._predict_call_count = 0
        self._embedding_time_total = 0.0
        self._cache_hits = 0
        self._cache_misses = 0

    def search_most_recent_jump_or_jumpi(self, curr_bytecode_index):
        i = curr_bytecode_index
        while (i < len(self.real_bytecode)):
            if self.real_bytecode[i] == "JUMPI" or self.real_bytecode[i] == "JUMP":
                return i
            i += 1
        return None
    
    def get_jumpSeq(self, curr_bytecode_index):
        jump_or_jumpi_index = self.search_most_recent_jump_or_jumpi(curr_bytecode_index)
        if jump_or_jumpi_index == None:
            return "Jump sequence is empty"
        else:
            # Capture 5 preceding bytecodes + the JUMP/JUMPI itself (6 tokens total)
            start = max(0, jump_or_jumpi_index - 5)
            seq_tokens = self.real_bytecode[start:jump_or_jumpi_index + 1]
            jumpSeq = " ".join(seq_tokens)
            return jumpSeq


    def count_smart_contract_jump_jumpi_number(self, real_bytecode):
        sum = 0
        for bytecode in real_bytecode:
            if bytecode == "JUMPI" or bytecode == "JUMP":
                sum += 1
        return sum

    def find_successor_count(self, bytecode_list_index):
        terminal_opcodes = {
            'JUMPI': 2,
            'JUMP': 1,
            'RETURN': 0,
            'STOP': 0,
            'REVERT': 0,
            'SELFDESTRUCT': 0
        }

        current_index = bytecode_list_index

        while current_index < len(self.real_bytecode):
            opcode = self.real_bytecode[current_index]

            if opcode in terminal_opcodes:
                successor_count = terminal_opcodes[opcode]
                return successor_count
            
            current_index += 1


        return -1

    def count_branch_new_instruction(self, index):
        terminal_opcodes = ['JUMPI', 'JUMP','RETURN', 'STOP', 'REVERT', 'SELFDESTRUCT']

        current_index = index

        while current_index < len(self.real_bytecode):
            if self.real_bytecode[current_index] in terminal_opcodes:
                temp_list = [index, current_index]
                # print("****************************************")
                # print(f"temp_list:{temp_list}")
                # print(f"self.passed_program_paths:{self.passed_program_paths}")
                # print("****************************************")
                for item in self.passed_program_paths:
                    if temp_list[1] == item[1]:
                        if temp_list[0] < item[0]:
                            return item[0] - temp_list[0], [temp_list[0], item[0] - 1]
                        else:
                            return 0, [-1, -1]
                return current_index - index + 1, [index, current_index]
            current_index += 1

    def count_cpicnt(self, index):
        terminal_opcodes = ['JUMPI', 'JUMP','RETURN', 'STOP', 'REVERT', 'SELFDESTRUCT']

        current_index = index

        while current_index < len(self.real_bytecode):
            if self.real_bytecode[current_index] in terminal_opcodes:
                return current_index - index + 1
            current_index += 1
    
    def count_execution_range(self, index):
        terminal_opcodes = ['JUMPI', 'JUMP','RETURN', 'STOP', 'REVERT', 'SELFDESTRUCT']

        current_index = index

        while current_index < len(self.real_bytecode):
            if self.real_bytecode[current_index] in terminal_opcodes:
                return [index, current_index]
            current_index += 1
    
    def count_covNew(self, node):
        p = node
        sum = 0
        while p != None and p.parent_node != None:
            temp_list = self.count_execution_range(p.parent_node.bytecode_list_index)
            if p.parent_node.branch_new_instruction == 0:
                sum += (temp_list[1] - temp_list[0] + 1)
                p = p.parent_node
            else:

                sum += (temp_list[1] - p.parent_node.branch_new_instruction_pc_range[1])
                break
        return sum
    
    def update_subpath_k4(self, node):
        k = 0
        p = node
        k4_list = []
        signal = True
        while k < 4:
            if p != None:
                k4_list.append(self.count_execution_range(p.bytecode_list_index))
            else:
                signal = False
                break
            p = p.parent_node
            k += 1
            
        if signal:
            k4_list.reverse()
            k4_list = f"{k4_list}"
            if k4_list in self.subpath_k4_to_number:
                self.subpath_k4_to_number[k4_list] += 1
            else:
                self.subpath_k4_to_number[k4_list] = 1
    
    def count_subpath_k4(self, node):
        k = 0
        p = node
        k4_list = []
        signal = True
        while k < 4:
            if p != None:
                k4_list.append(self.count_execution_range(p.bytecode_list_index))
            else:
                signal = False
                break
            p = p.parent_node
            k += 1
            
        if signal:
            k4_list.reverse()
            k4_list = f"{k4_list}"
            if k4_list in self.subpath_k4_to_number:
                return self.subpath_k4_to_number[k4_list]
            else:
                return 0
        else:
            return 0
    
    def prue_tree(self, node):

        new_children = []
        for child in node.children_node:
            prued_child = self.prue_tree(child)
            if prued_child is not None:
                new_children.append(prued_child)
        node.children_node = new_children


        if not node.executed:
            return None
        else:
            return node

    def count_node_reward(self, node):
        sum = 0
        if len(node.children_node) == 0:
            node.reward = node.branch_new_block + node.path_new_block
            return node.branch_new_block + node.path_new_block
        for child_node in node.children_node:
            sum += self.count_node_reward(child_node)
        node.reward = sum
        return sum

    def count_select_state_accuracy(self, select_state_accuracy_count):
        sum = 0
        for item in select_state_accuracy_count:
            if item == 1:
                sum += 1
        return sum / len(select_state_accuracy_count)


    def execute(self):
        print(self.strategy)
        if (self.strategy == "rss"):
            self.dispatcher_boundary = self.get_max_stop_return_index()
            if self.dispatcher_boundary == -1:
                return False
            self.execution_paths = [(self.bytecode_list_index, self.stack.copy())] # !
            self.temp_nodes = [self.origin_node]
            all_stacks = []
            self.smartcontract_functions_index_position.append(self.dispatcher_boundary + 1)
            start_time = time.time()
            while self.execution_paths and time.time() - start_time < MAX_TIME:
                # self.bytecode_list_index, self.stack = self.execution_paths.pop()
                # print(f"self.execution_paths: {self.execution_paths}") # !
                rand_index = random.randint(0, len(self.execution_paths) - 1)


                for node in self.temp_nodes:
                    if node != self.origin_node:
                        node.test_case_number = self.test_case_num
                        node.branch_new_instruction, node.branch_new_instruction_pc_range = self.count_branch_new_instruction(node.bytecode_list_index)
                        if node.branch_new_instruction:
                            node.branch_new_block = 1
                        else:
                            node.branch_new_block = 0

                        if node.parent_node.is_occupied:
                            node.path_new_instruction = 0
                            node.path_new_block = 0
                        else:
                            node.path_new_instruction = node.parent_node.path_new_instruction + node.parent_node.branch_new_instruction
                            node.path_new_block = node.parent_node.path_new_block + node.parent_node.branch_new_block

                        passed_range = self.count_execution_range(node.bytecode_list_index)
                        passed_range = f"{passed_range}"
                        if passed_range in self.passed_program_paths_to_passed_number:
                            node.icnt = self.passed_program_paths_to_passed_number[passed_range]
                        else:
                            node.icnt = 0

                        node.subpath = self.count_subpath_k4(node)
    

                self.temp_node = self.temp_nodes.pop(rand_index)
                if (self.temp_node.branch_new_instruction != 0):
                    self.select_state_accuracy_count.append(1)
                else:
                    self.select_state_accuracy_count.append(0)

                self.temp_node.executed = True

                if node != self.origin_node:
                    self.temp_node.parent_node.is_occupied = True

                self.bytecode_list_index, self.stack = self.execution_paths.pop(rand_index)
                self.passed_program_paths.append([self.bytecode_list_index, -1])
                if self.detector:
                    self.detector.reset_path()


                #     self.test_case_num += 1
                    
                while self.bytecode_list_index < len(self.real_bytecode):
                    # print(f"index:{self.bytecode_list_index}")
                    # print(f"stack:{self.stack}")
                    opcode = self.symbolic_bytecode[self.bytecode_list_index] # !
                    # print(opcode)
                    handler_name = opcode.lower()
                    handler = getattr(self.handlers, handler_name, None)
                    self.record_stack_snapshot()
                    self.record_stack()

                    # Vulnerability detection callback (inference phase only)
                    if self.detector:
                        self.detector.on_opcode(opcode, self.bytecode_list_index, self.stack)

                    if opcode.lower().startswith("push") and opcode.lower() != "push0":
                        value = self.symbolic_bytecode[self.bytecode_list_index + 1] # !
                        handler(value)
                    elif opcode.lower() == "jump":
                        jump_address_symbol = self.stack[-1] # !


                        if "&" not in str(jump_address_symbol):
                            jump_address_pc = name_to_value[jump_address_symbol]
                        else:
                            parts = str(jump_address_symbol).split("&")

                            result = parts[1].strip()
                            jump_address_pc = name_to_value[BitVec(result, 256)]

                        jump_address_index = self.get_index_position(jump_address_pc)
                        # print(name_to_value)
                        # print(f"jump address pc: {self.stack[-1]}")
                        # print(type(self.stack[-1]))
                        # print(f"jump address pc: {jump_address_pc}")
                        # print(f"jump address index: {jump_address_index}")


                        if jump_address_index <= self.dispatcher_boundary:

                            if self.bytecode_list_index + 1 > self.dispatcher_boundary + 2: # !
                                if (
                                    self.bytecode_list_index + 1 # !
                                    not in self.smartcontract_functions_index_position
                                ):
                                    self.smartcontract_functions_index_position.append(
                                        self.bytecode_list_index + 1 # !
                                    )
                                    # print(
                                    #     f"self.smartcontract_functions_index_position: {self.smartcontract_functions_index_position}"
                                    # )

                        handler(jump_address_index)
                        break
                        # exist another way
                    elif opcode.lower() == "jumpi":
                        # print(f"happen branch jumpi in index {self.bytecode_list_index}")
                        jumpi_address_symbol = self.stack[-1] # !
                        jumpi_condition = self.stack[-2] # !
                        jumpi_address_pc = name_to_value[jumpi_address_symbol]
                        jumpi_address_index = self.get_index_position(jumpi_address_pc)









                        if self.bytecode_list_index in self.visited_nodes_index_by_jumpi: # !

                            if (
                                self.visited_nodes_index_by_jumpi[self.bytecode_list_index] # !
                                <= 5
                            ):
                                self.visited_nodes_index_by_jumpi[
                                    self.bytecode_list_index # !
                                ] += 1
                                handler(jumpi_address_index, jumpi_condition)
                            else:
                                # print(
                                #     f"Exist Loop!!! Exist Loop!!! Exist Loop!!! in PC {self.bytecode_list_index}"
                                # )

                                self.exist_loop_node_by_jumpi.add(self.bytecode_list_index) # !


                                # self.test_case_num += 1

                                handler(jumpi_address_index, jumpi_condition)
                        else:
                            self.visited_nodes_index_by_jumpi.update(
                                {self.bytecode_list_index: 0} # !
                            )
                            handler(jumpi_address_index, jumpi_condition)

                        break
                    elif (
                        opcode.lower() == "return"
                    ):
                        handler_name = "return_op"
                        handler = getattr(self.handlers, handler_name, None)
                        handler()
                    elif opcode.lower().startswith("and"):  # 。。。
                        handler_name = "and_op"
                        handler = getattr(self.handlers, handler_name, None)
                        handler()
                    elif opcode.lower() == "or":  # 。。。
                        handler_name = "or_op"
                        handler = getattr(self.handlers, handler_name, None)
                        handler()
                    elif opcode.lower() == "xor":  # 。。。
                        handler_name = "xor_op"
                        handler = getattr(self.handlers, handler_name, None)
                        handler()
                    elif opcode.lower() == "not":  # 。。。
                        handler_name = "not_op"
                        handler = getattr(self.handlers, handler_name, None)
                        handler()
                    elif opcode.lower() == "byte":  # 。。。
                        handler_name = "byte_op"
                        handler = getattr(self.handlers, handler_name, None)
                        handler()
                    else:
                        handler()

                    if handler is None:
                        raise ValueError(f"Unknown opcode: {opcode}")

                all_stacks.append(self.stack.copy()) # !

                self.coverage = len(self.control_flow_graph) / self.all_jump_jumpi_number
                # print(f"self.coverage:{self.coverage}")
                if (self.coverage > self.assigned_coverage and len(self.arrive_assigned_coverage_time) == 0):
                    self.arrive_assigned_coverage_time.append(time.time() - start_time)
                # print(f"self.arrive_assigned_coverage_time:{self.arrive_assigned_coverage_time}")

                # print("***************************************************")
                # print(self.temp_node.branch_new_instruction)
                # print("***************************************************")

            print(f"self.coverage:{self.coverage}")
            print(f"self.arrive_assigned_coverage_time:{self.arrive_assigned_coverage_time}")
            # print(f"self.select_state_accuracy_count:{self.select_state_accuracy_count}")
            self.select_state_accuracy = self.count_select_state_accuracy(self.select_state_accuracy_count)
            print(f"self.select_state_accuracy:{self.select_state_accuracy}")

            self.control_flow_graph = dict(sorted(self.control_flow_graph.items()))
            print(f"self.control_flow_graph: {self.control_flow_graph}")
        elif isinstance(self.strategy, list) and self.strategy[0] == "learch":
            self.dispatcher_boundary = self.get_max_stop_return_index()
            if self.dispatcher_boundary == -1:
                return False
            self.execution_paths = [(self.bytecode_list_index, self.stack.copy())] # !
            self.temp_nodes = [self.origin_node]
            all_stacks = []
            self.smartcontract_functions_index_position.append(self.dispatcher_boundary + 1)
            start_time = time.time()
            while self.execution_paths and time.time() - start_time < MAX_TIME:
                # self.bytecode_list_index, self.stack = self.execution_paths.pop()
                # print(f"self.execution_paths: {self.execution_paths}") # !



                for node in self.temp_nodes:
                    if node != self.origin_node:
                        node.test_case_number = self.test_case_num
                        node.branch_new_instruction, node.branch_new_instruction_pc_range = self.count_branch_new_instruction(node.bytecode_list_index)
                        if node.branch_new_instruction:
                            node.branch_new_block = 1
                        else:
                            node.branch_new_block = 0

                        if node.parent_node.is_occupied:
                            node.path_new_instruction = 0
                            node.path_new_block = 0
                        else:
                            node.path_new_instruction = node.parent_node.path_new_instruction + node.parent_node.branch_new_instruction
                            node.path_new_block = node.parent_node.path_new_block + node.parent_node.branch_new_block

                        passed_range = self.count_execution_range(node.bytecode_list_index)
                        passed_range = f"{passed_range}"
                        if passed_range in self.passed_program_paths_to_passed_number:
                            node.icnt = self.passed_program_paths_to_passed_number[passed_range]
                        else:
                            node.icnt = 0

                        node.subpath = self.count_subpath_k4(node)
    

                reward_max_index = 0
                reward_max = 0
                i = 0
                while (i < len(self.temp_nodes)):
                    temp_node = self.temp_nodes[i]
                    features = [{
                        "stack_size": temp_node.stack_size,
                        "successor_number": temp_node.successor_number,
                        "test_case_number": temp_node.test_case_number,
                        "branch_new_instruction": temp_node.branch_new_instruction,
                        "path_new_instruction": temp_node.path_new_instruction,
                        "depth": temp_node.depth,
                        "cpicnt": temp_node.cpicnt,
                        "icnt": temp_node.icnt,
                        "covNew": temp_node.covNew,
                        "subpath": temp_node.subpath,
                    },]
                    # Normalize
                    features[0]["stack_size"] = min(features[0]["stack_size"] / STACK_MAX, 1)
                    features[0]["successor_number"] = min(features[0]["successor_number"] / SUCCESSOR_MAX, 1)
                    features[0]["test_case_number"] = min(features[0]["test_case_number"] / TEST_CASE_NUMBER_MAX, 1)
                    features[0]["branch_new_instruction"] = min(features[0]["branch_new_instruction"] / len(self.real_bytecode), 1)
                    features[0]["path_new_instruction"] = min(features[0]["path_new_instruction"] / len(self.real_bytecode), 1)
                    features[0]["depth"] = min(features[0]["depth"] / DEPTH_MAX, 1)
                    features[0]["cpicnt"] = min(features[0]["cpicnt"] / len(self.real_bytecode), 1)
                    features[0]["icnt"] = min(features[0]["icnt"] / ICNT_MAX, 1)
                    features[0]["covNew"] = min(features[0]["covNew"] / len(self.real_bytecode), 1)
                    features[0]["subpath"] = min(features[0]["subpath"] / SUBPATH_MAX, 1)

                    _t_pred = time.time()
                    reward = self.strategy[1].predict(features)[0]
                    self._predict_time_total += time.time() - _t_pred
                    self._predict_call_count += 1
                    if (reward > reward_max):
                        reward_max = reward
                        reward_max_index = i
                    
                    i += 1

                self.temp_node = self.temp_nodes.pop(reward_max_index)
                if (self.temp_node.branch_new_instruction != 0):
                    self.select_state_accuracy_count.append(1)
                else:
                    self.select_state_accuracy_count.append(0)

                self.temp_node.executed = True

                if node != self.origin_node:
                    self.temp_node.parent_node.is_occupied = True

                self.bytecode_list_index, self.stack = self.execution_paths.pop(reward_max_index)
                self.passed_program_paths.append([self.bytecode_list_index, -1])
                if self.detector:
                    self.detector.reset_path()


                #     self.test_case_num += 1
                    
                while self.bytecode_list_index < len(self.real_bytecode):
                    # print(f"index:{self.bytecode_list_index}")
                    # print(f"stack:{self.stack}")
                    opcode = self.symbolic_bytecode[self.bytecode_list_index] # !
                    # print(opcode)
                    handler_name = opcode.lower()
                    handler = getattr(self.handlers, handler_name, None)
                    self.record_stack_snapshot()
                    self.record_stack()

                    # Vulnerability detection callback (inference phase only)
                    if self.detector:
                        self.detector.on_opcode(opcode, self.bytecode_list_index, self.stack)

                    if opcode.lower().startswith("push") and opcode.lower() != "push0":
                        value = self.symbolic_bytecode[self.bytecode_list_index + 1] # !
                        handler(value)
                    elif opcode.lower() == "jump":
                        jump_address_symbol = self.stack[-1] # !


                        if "&" not in str(jump_address_symbol):
                            jump_address_pc = name_to_value[jump_address_symbol]
                        else:
                            parts = str(jump_address_symbol).split("&")

                            result = parts[1].strip()
                            jump_address_pc = name_to_value[BitVec(result, 256)]

                        jump_address_index = self.get_index_position(jump_address_pc)
                        # print(name_to_value)
                        # print(f"jump address pc: {self.stack[-1]}")
                        # print(type(self.stack[-1]))
                        # print(f"jump address pc: {jump_address_pc}")
                        # print(f"jump address index: {jump_address_index}")


                        if jump_address_index <= self.dispatcher_boundary:

                            if self.bytecode_list_index + 1 > self.dispatcher_boundary + 2: # !
                                if (
                                    self.bytecode_list_index + 1 # !
                                    not in self.smartcontract_functions_index_position
                                ):
                                    self.smartcontract_functions_index_position.append(
                                        self.bytecode_list_index + 1 # !
                                    )
                                    # print(
                                    #     f"self.smartcontract_functions_index_position: {self.smartcontract_functions_index_position}"
                                    # )

                        handler(jump_address_index)
                        break
                        # exist another way
                    elif opcode.lower() == "jumpi":
                        # print(f"happen branch jumpi in index {self.bytecode_list_index}")
                        jumpi_address_symbol = self.stack[-1] # !
                        jumpi_condition = self.stack[-2] # !
                        jumpi_address_pc = name_to_value[jumpi_address_symbol]
                        jumpi_address_index = self.get_index_position(jumpi_address_pc)









                        if self.bytecode_list_index in self.visited_nodes_index_by_jumpi: # !

                            if (
                                self.visited_nodes_index_by_jumpi[self.bytecode_list_index] # !
                                <= 5
                            ):
                                self.visited_nodes_index_by_jumpi[
                                    self.bytecode_list_index # !
                                ] += 1
                                handler(jumpi_address_index, jumpi_condition)
                            else:
                                # print(
                                #     f"Exist Loop!!! Exist Loop!!! Exist Loop!!! in PC {self.bytecode_list_index}"
                                # )

                                self.exist_loop_node_by_jumpi.add(self.bytecode_list_index) # !


                                # self.test_case_num += 1

                                handler(jumpi_address_index, jumpi_condition)
                        else:
                            self.visited_nodes_index_by_jumpi.update(
                                {self.bytecode_list_index: 0} # !
                            )
                            handler(jumpi_address_index, jumpi_condition)

                        break
                    elif (
                        opcode.lower() == "return"
                    ):
                        handler_name = "return_op"
                        handler = getattr(self.handlers, handler_name, None)
                        handler()
                    elif opcode.lower().startswith("and"):  # 。。。
                        handler_name = "and_op"
                        handler = getattr(self.handlers, handler_name, None)
                        handler()
                    elif opcode.lower() == "or":  # 。。。
                        handler_name = "or_op"
                        handler = getattr(self.handlers, handler_name, None)
                        handler()
                    elif opcode.lower() == "xor":  # 。。。
                        handler_name = "xor_op"
                        handler = getattr(self.handlers, handler_name, None)
                        handler()
                    elif opcode.lower() == "not":  # 。。。
                        handler_name = "not_op"
                        handler = getattr(self.handlers, handler_name, None)
                        handler()
                    elif opcode.lower() == "byte":  # 。。。
                        handler_name = "byte_op"
                        handler = getattr(self.handlers, handler_name, None)
                        handler()
                    else:
                        handler()

                    if handler is None:
                        raise ValueError(f"Unknown opcode: {opcode}")

                all_stacks.append(self.stack.copy()) # !

                self.coverage = len(self.control_flow_graph) / self.all_jump_jumpi_number
                # print(f"self.coverage:{self.coverage}")
                if (self.coverage > self.assigned_coverage and len(self.arrive_assigned_coverage_time) == 0):
                    self.arrive_assigned_coverage_time.append(time.time() - start_time)
                # print(f"self.arrive_assigned_coverage_time:{self.arrive_assigned_coverage_time}")

                # print("***************************************************")
                # print(self.temp_node.branch_new_instruction)
                # print("***************************************************")
            
            print(f"self.coverage:{self.coverage}")
            print(f"self.arrive_assigned_coverage_time:{self.arrive_assigned_coverage_time}")
            # print(f"self.select_state_accuracy_count:{self.select_state_accuracy_count}")
            self.select_state_accuracy = self.count_select_state_accuracy(self.select_state_accuracy_count)
            print(f"self.select_state_accuracy:{self.select_state_accuracy}")

            self.control_flow_graph = dict(sorted(self.control_flow_graph.items()))
            print(f"self.control_flow_graph: {self.control_flow_graph}")
        elif hasattr(self.strategy, 'select'):  # Pluggable baseline strategy (BaseStrategy subclass)
            self.dispatcher_boundary = self.get_max_stop_return_index()
            if self.dispatcher_boundary == -1:
                return False
            self.execution_paths = [(self.bytecode_list_index, self.stack.copy())]
            self.temp_nodes = [self.origin_node]
            all_stacks = []
            self.smartcontract_functions_index_position.append(self.dispatcher_boundary + 1)
            start_time = time.time()
            while self.execution_paths and time.time() - start_time < MAX_TIME:

                # Dynamic feature update (same as learch branch)
                for node in self.temp_nodes:
                    if node != self.origin_node:
                        node.test_case_number = self.test_case_num
                        node.branch_new_instruction, node.branch_new_instruction_pc_range = self.count_branch_new_instruction(node.bytecode_list_index)
                        if node.branch_new_instruction:
                            node.branch_new_block = 1
                        else:
                            node.branch_new_block = 0
                        if node.parent_node.is_occupied:
                            node.path_new_instruction = 0
                            node.path_new_block = 0
                        else:
                            node.path_new_instruction = node.parent_node.path_new_instruction + node.parent_node.branch_new_instruction
                            node.path_new_block = node.parent_node.path_new_block + node.parent_node.branch_new_block
                        passed_range = self.count_execution_range(node.bytecode_list_index)
                        passed_range = f"{passed_range}"
                        if passed_range in self.passed_program_paths_to_passed_number:
                            node.icnt = self.passed_program_paths_to_passed_number[passed_range]
                        else:
                            node.icnt = 0
                        node.subpath = self.count_subpath_k4(node)

                # --- Baseline strategy selects the state ---
                context = {
                    "control_flow_graph": self.control_flow_graph,
                    "real_bytecode_len": len(self.real_bytecode),
                    "all_jump_jumpi_number": self.all_jump_jumpi_number,
                    "STACK_MAX": STACK_MAX,
                    "SUCCESSOR_MAX": SUCCESSOR_MAX,
                    "TEST_CASE_NUMBER_MAX": TEST_CASE_NUMBER_MAX,
                    "DEPTH_MAX": DEPTH_MAX,
                    "ICNT_MAX": ICNT_MAX,
                    "SUBPATH_MAX": SUBPATH_MAX,
                    "real_bytecode": self.real_bytecode,
                    "dispatcher_boundary": self.dispatcher_boundary,
                    "smartcontract_functions_index_position": self.smartcontract_functions_index_position,
                }
                selected_index = self.strategy.select(self.temp_nodes, context)

                self.temp_node = self.temp_nodes.pop(selected_index)
                if (self.temp_node.branch_new_instruction != 0):
                    self.select_state_accuracy_count.append(1)
                else:
                    self.select_state_accuracy_count.append(0)

                self.temp_node.executed = True

                if node != self.origin_node:
                    self.temp_node.parent_node.is_occupied = True

                self.bytecode_list_index, self.stack = self.execution_paths.pop(selected_index)
                self.passed_program_paths.append([self.bytecode_list_index, -1])
                if self.detector:
                    self.detector.reset_path()

                while self.bytecode_list_index < len(self.real_bytecode):
                    opcode = self.symbolic_bytecode[self.bytecode_list_index]
                    handler_name = opcode.lower()
                    handler = getattr(self.handlers, handler_name, None)
                    self.record_stack_snapshot()
                    self.record_stack()

                    if self.detector:
                        self.detector.on_opcode(opcode, self.bytecode_list_index, self.stack)

                    if opcode.lower().startswith("push") and opcode.lower() != "push0":
                        value = self.symbolic_bytecode[self.bytecode_list_index + 1]
                        handler(value)
                    elif opcode.lower() == "jump":
                        jump_address_symbol = self.stack[-1]
                        if "&" not in str(jump_address_symbol):
                            jump_address_pc = name_to_value[jump_address_symbol]
                        else:
                            parts = str(jump_address_symbol).split("&")
                            result = parts[1].strip()
                            jump_address_pc = name_to_value[BitVec(result, 256)]
                        jump_address_index = self.get_index_position(jump_address_pc)
                        if jump_address_index <= self.dispatcher_boundary:
                            if self.bytecode_list_index + 1 > self.dispatcher_boundary + 2:
                                if (self.bytecode_list_index + 1 not in self.smartcontract_functions_index_position):
                                    self.smartcontract_functions_index_position.append(self.bytecode_list_index + 1)
                        handler(jump_address_index)
                        break
                    elif opcode.lower() == "jumpi":
                        jumpi_address_symbol = self.stack[-1]
                        jumpi_condition = self.stack[-2]
                        jumpi_address_pc = name_to_value[jumpi_address_symbol]
                        jumpi_address_index = self.get_index_position(jumpi_address_pc)
                        if self.bytecode_list_index in self.visited_nodes_index_by_jumpi:
                            if (self.visited_nodes_index_by_jumpi[self.bytecode_list_index] <= 5):
                                self.visited_nodes_index_by_jumpi[self.bytecode_list_index] += 1
                                handler(jumpi_address_index, jumpi_condition)
                            else:
                                self.exist_loop_node_by_jumpi.add(self.bytecode_list_index)
                                handler(jumpi_address_index, jumpi_condition)
                        else:
                            self.visited_nodes_index_by_jumpi.update({self.bytecode_list_index: 0})
                            handler(jumpi_address_index, jumpi_condition)
                        break
                    elif opcode.lower() == "return":
                        handler_name = "return_op"
                        handler = getattr(self.handlers, handler_name, None)
                        handler()
                    elif opcode.lower().startswith("and"):
                        handler_name = "and_op"
                        handler = getattr(self.handlers, handler_name, None)
                        handler()
                    elif opcode.lower() == "or":
                        handler_name = "or_op"
                        handler = getattr(self.handlers, handler_name, None)
                        handler()
                    elif opcode.lower() == "xor":
                        handler_name = "xor_op"
                        handler = getattr(self.handlers, handler_name, None)
                        handler()
                    elif opcode.lower() == "not":
                        handler_name = "not_op"
                        handler = getattr(self.handlers, handler_name, None)
                        handler()
                    elif opcode.lower() == "byte":
                        handler_name = "byte_op"
                        handler = getattr(self.handlers, handler_name, None)
                        handler()
                    else:
                        handler()

                    if handler is None:
                        raise ValueError(f"Unknown opcode: {opcode}")

                all_stacks.append(self.stack.copy())

                self.coverage = len(self.control_flow_graph) / self.all_jump_jumpi_number
                if (self.coverage > self.assigned_coverage and len(self.arrive_assigned_coverage_time) == 0):
                    self.arrive_assigned_coverage_time.append(time.time() - start_time)

            print(f"self.coverage:{self.coverage}")
            print(f"self.arrive_assigned_coverage_time:{self.arrive_assigned_coverage_time}")
            self.select_state_accuracy = self.count_select_state_accuracy(self.select_state_accuracy_count)
            print(f"self.select_state_accuracy:{self.select_state_accuracy}")

            self.control_flow_graph = dict(sorted(self.control_flow_graph.items()))
            print(f"self.control_flow_graph: {self.control_flow_graph}")
        else:
            self.dispatcher_boundary = self.get_max_stop_return_index()
            if self.dispatcher_boundary == -1:
                return False
            self.execution_paths = [(self.bytecode_list_index, self.stack.copy())] # !
            self.temp_nodes = [self.origin_node]
            all_stacks = []
            self.smartcontract_functions_index_position.append(self.dispatcher_boundary + 1)
            start_time = time.time()
            while self.execution_paths and time.time() - start_time < MAX_TIME:
                # self.bytecode_list_index, self.stack = self.execution_paths.pop()
                # print(f"self.execution_paths: {self.execution_paths}") # !



                for node in self.temp_nodes:
                    if node != self.origin_node:
                        node.test_case_number = self.test_case_num
                        node.branch_new_instruction, node.branch_new_instruction_pc_range = self.count_branch_new_instruction(node.bytecode_list_index)
                        if node.branch_new_instruction:
                            node.branch_new_block = 1
                        else:
                            node.branch_new_block = 0

                        if node.parent_node.is_occupied:
                            node.path_new_instruction = 0
                            node.path_new_block = 0
                        else:
                            node.path_new_instruction = node.parent_node.path_new_instruction + node.parent_node.branch_new_instruction
                            node.path_new_block = node.parent_node.path_new_block + node.parent_node.branch_new_block

                        passed_range = self.count_execution_range(node.bytecode_list_index)
                        passed_range = f"{passed_range}"
                        if passed_range in self.passed_program_paths_to_passed_number:
                            node.icnt = self.passed_program_paths_to_passed_number[passed_range]
                        else:
                            node.icnt = 0

                        node.subpath = self.count_subpath_k4(node)
    

                reward_max_index = 0
                reward_max = 0
                i = 0
                while (i < len(self.temp_nodes)):
                    temp_node = self.temp_nodes[i]

                    features_1 = [temp_node.stack_size, temp_node.successor_number, temp_node.test_case_number, temp_node.branch_new_instruction, temp_node.path_new_instruction, temp_node.depth, temp_node.cpicnt, temp_node.icnt, temp_node.covNew, temp_node.subpath]
                    # Normalize
                    features_1[0] = min(features_1[0] / STACK_MAX, 1)
                    features_1[1] = min(features_1[1] / SUCCESSOR_MAX, 1)
                    features_1[2] = min(features_1[2] / TEST_CASE_NUMBER_MAX, 1)
                    features_1[3] = min(features_1[3] / len(self.real_bytecode), 1)
                    features_1[4] = min(features_1[4] / len(self.real_bytecode), 1)
                    features_1[5] = min(features_1[5] / DEPTH_MAX, 1)
                    features_1[6] = min(features_1[6] / len(self.real_bytecode), 1)
                    features_1[7] = min(features_1[7] / ICNT_MAX, 1)
                    features_1[8] = min(features_1[8] / len(self.real_bytecode), 1)
                    features_1[9] = min(features_1[9] / SUBPATH_MAX, 1)


                    features_2 = [temp_node.jumpSeq, temp_node.bytecode_list_index, temp_node.constraint]

                    _t_pred = time.time()
                    reward = self.strategy[1].predict(features_1, features_2)
                    _pred_elapsed = time.time() - _t_pred
                    self._predict_time_total += _pred_elapsed
                    self._predict_call_count += 1
                    self._embedding_time_total += getattr(self.strategy[1], '_last_embedding_time', 0)
                    if getattr(self.strategy[1], '_last_embedding_time', 0) > 0:
                        from feature_fusion import symflow_feature_fusion as _sff
                        if getattr(_sff, '_last_was_cached', False):
                            self._cache_hits += 1
                        else:
                            self._cache_misses += 1
                    if (reward > reward_max):
                        reward_max = reward
                        reward_max_index = i
                    
                    i += 1

                self.temp_node = self.temp_nodes.pop(reward_max_index)
                if (self.temp_node.branch_new_instruction != 0):
                    self.select_state_accuracy_count.append(1)
                else:
                    self.select_state_accuracy_count.append(0)

                self.temp_node.executed = True

                if node != self.origin_node:
                    self.temp_node.parent_node.is_occupied = True

                self.bytecode_list_index, self.stack = self.execution_paths.pop(reward_max_index)
                self.passed_program_paths.append([self.bytecode_list_index, -1])
                if self.detector:
                    self.detector.reset_path()


                #     self.test_case_num += 1
                    
                while self.bytecode_list_index < len(self.real_bytecode):
                    # print(f"index:{self.bytecode_list_index}")
                    # print(f"stack:{self.stack}")
                    opcode = self.symbolic_bytecode[self.bytecode_list_index] # !
                    # print(opcode)
                    handler_name = opcode.lower()
                    handler = getattr(self.handlers, handler_name, None)
                    self.record_stack_snapshot()
                    self.record_stack()

                    # Vulnerability detection callback (inference phase only)
                    if self.detector:
                        self.detector.on_opcode(opcode, self.bytecode_list_index, self.stack)

                    if opcode.lower().startswith("push") and opcode.lower() != "push0":
                        value = self.symbolic_bytecode[self.bytecode_list_index + 1] # !
                        handler(value)
                    elif opcode.lower() == "jump":
                        jump_address_symbol = self.stack[-1] # !


                        if "&" not in str(jump_address_symbol):
                            jump_address_pc = name_to_value[jump_address_symbol]
                        else:
                            parts = str(jump_address_symbol).split("&")

                            result = parts[1].strip()
                            jump_address_pc = name_to_value[BitVec(result, 256)]

                        jump_address_index = self.get_index_position(jump_address_pc)
                        # print(name_to_value)
                        # print(f"jump address pc: {self.stack[-1]}")
                        # print(type(self.stack[-1]))
                        # print(f"jump address pc: {jump_address_pc}")
                        # print(f"jump address index: {jump_address_index}")


                        if jump_address_index <= self.dispatcher_boundary:

                            if self.bytecode_list_index + 1 > self.dispatcher_boundary + 2: # !
                                if (
                                    self.bytecode_list_index + 1 # !
                                    not in self.smartcontract_functions_index_position
                                ):
                                    self.smartcontract_functions_index_position.append(
                                        self.bytecode_list_index + 1 # !
                                    )
                                    # print(
                                    #     f"self.smartcontract_functions_index_position: {self.smartcontract_functions_index_position}"
                                    # )

                        handler(jump_address_index)
                        break
                        # exist another way
                    elif opcode.lower() == "jumpi":
                        # print(f"happen branch jumpi in index {self.bytecode_list_index}")
                        jumpi_address_symbol = self.stack[-1] # !
                        jumpi_condition = self.stack[-2] # !
                        jumpi_address_pc = name_to_value[jumpi_address_symbol]
                        jumpi_address_index = self.get_index_position(jumpi_address_pc)









                        if self.bytecode_list_index in self.visited_nodes_index_by_jumpi: # !

                            if (
                                self.visited_nodes_index_by_jumpi[self.bytecode_list_index] # !
                                <= 5
                            ):
                                self.visited_nodes_index_by_jumpi[
                                    self.bytecode_list_index # !
                                ] += 1
                                handler(jumpi_address_index, jumpi_condition)
                            else:
                                # print(
                                #     f"Exist Loop!!! Exist Loop!!! Exist Loop!!! in PC {self.bytecode_list_index}"
                                # )

                                self.exist_loop_node_by_jumpi.add(self.bytecode_list_index) # !


                                # self.test_case_num += 1

                                handler(jumpi_address_index, jumpi_condition)
                        else:
                            self.visited_nodes_index_by_jumpi.update(
                                {self.bytecode_list_index: 0} # !
                            )
                            handler(jumpi_address_index, jumpi_condition)

                        break
                    elif (
                        opcode.lower() == "return"
                    ):
                        handler_name = "return_op"
                        handler = getattr(self.handlers, handler_name, None)
                        handler()
                    elif opcode.lower().startswith("and"):  # 。。。
                        handler_name = "and_op"
                        handler = getattr(self.handlers, handler_name, None)
                        handler()
                    elif opcode.lower() == "or":  # 。。。
                        handler_name = "or_op"
                        handler = getattr(self.handlers, handler_name, None)
                        handler()
                    elif opcode.lower() == "xor":  # 。。。
                        handler_name = "xor_op"
                        handler = getattr(self.handlers, handler_name, None)
                        handler()
                    elif opcode.lower() == "not":  # 。。。
                        handler_name = "not_op"
                        handler = getattr(self.handlers, handler_name, None)
                        handler()
                    elif opcode.lower() == "byte":  # 。。。
                        handler_name = "byte_op"
                        handler = getattr(self.handlers, handler_name, None)
                        handler()
                    else:
                        handler()

                    if handler is None:
                        raise ValueError(f"Unknown opcode: {opcode}")

                all_stacks.append(self.stack.copy()) # !

                self.coverage = len(self.control_flow_graph) / self.all_jump_jumpi_number
                # print(f"self.coverage:{self.coverage}")
                if (self.coverage > self.assigned_coverage and len(self.arrive_assigned_coverage_time) == 0):
                    self.arrive_assigned_coverage_time.append(time.time() - start_time)
                # print(f"self.arrive_assigned_coverage_time:{self.arrive_assigned_coverage_time}")

                # print("***************************************************")
                # print(self.temp_node.branch_new_instruction)
                # print("***************************************************")
            
            print(f"self.coverage:{self.coverage}")
            print(f"self.arrive_assigned_coverage_time:{self.arrive_assigned_coverage_time}")
            # print(f"self.select_state_accuracy_count:{self.select_state_accuracy_count}")
            self.select_state_accuracy = self.count_select_state_accuracy(self.select_state_accuracy_count)
            print(f"self.select_state_accuracy:{self.select_state_accuracy}")

            self.control_flow_graph = dict(sorted(self.control_flow_graph.items()))
            print(f"self.control_flow_graph: {self.control_flow_graph}")
        # elif (self.strategy != "rss" and self.strategy!= "rps" and self.strategy!= "nurs" and self.strategy!= "sgs"):
        #     pass

        return True

    def clarify_function_information(self):



        all_target_jumpdest_index = []
        for key in self.control_flow_graph.keys():
            all_target_jumpdest_index.extend(self.control_flow_graph[key])
        all_target_jumpdest_index = set(all_target_jumpdest_index)
        all_target_jumpdest_index = list(all_target_jumpdest_index)
        all_target_jumpdest_index = sorted(all_target_jumpdest_index)
        print(f"all_target_jumpdest_index: {all_target_jumpdest_index}")

        exist_internal_function_boundrys_index = set()
        index = self.dispatcher_boundary + 1
        while index <= self.the_first_contract_end_index:
            if (
                self.real_bytecode[index] == "JUMP"
                and self.real_bytecode[index - 2].startswith("PUSH")
                and self.real_bytecode[index - 2] != "PUSH0"
                and self.real_bytecode[index + 1] == "JUMPDEST"
            ):
                print(f"new turn!!!")
                repeated_jump_nodes = set()
                passed_paths_range_list = []
                possible_search_index = []
                possible_search_index.append(index)
                while possible_search_index:
                    current_index = possible_search_index.pop()
                    current_index_target_address_index_list = self.control_flow_graph[
                        current_index
                    ]
                    print(
                        f"current_index_target_address_index_list: {current_index_target_address_index_list}"
                    )
                    if current_index in repeated_jump_nodes:
                        continue
                    else:
                        repeated_jump_nodes.add(current_index)

                    for target_address_index in current_index_target_address_index_list:
                        keep_go_signal = True
                        next_jump_or_jumpi_index = min(
                            [
                                key
                                for key in self.control_flow_graph.keys()
                                if key > target_address_index
                            ]
                        )
                        for index1 in range(
                            target_address_index + 1, next_jump_or_jumpi_index
                        ):
                            if (
                                self.real_bytecode[index1] == "RETURN"
                                or self.real_bytecode[index1] == "STOP"
                                or self.real_bytecode[index1] == "REVERT"
                                or self.real_bytecode[index1] == "INVALID"
                                or self.real_bytecode[index1] == "SELFDESTRUCT"
                            ):
                                keep_go_signal = False
                                break
                        if keep_go_signal:
                            possible_search_index.append(next_jump_or_jumpi_index)
                            passed_paths_range_list.append(
                                [target_address_index, next_jump_or_jumpi_index]
                            )
                    print(f"passed_paths_range_list: {passed_paths_range_list}")
                print(f"repeated_jump_nodes: {repeated_jump_nodes}")
                for path in passed_paths_range_list:
                    if index + 1 in range(path[0], path[1] + 1):
                        print("111")
                        print(f"index+1: {index + 1}")
                        for key in self.control_flow_graph.keys():
                            if (
                                index + 1 in self.control_flow_graph[key]
                                and self.real_bytecode[key] == "JUMP"
                            ):
                                print("222")
                                self.all_jump_index_related_to_Call.add(index)
                                exist_internal_function_boundrys_index.add(key + 1)

            index += 1

        print(
            f"self.all_jump_index_related_to_Call: {self.all_jump_index_related_to_Call}"
        )
        print(
            f"exist_internal_function_boundrys_index: {exist_internal_function_boundrys_index}"
        )

        for index in list(exist_internal_function_boundrys_index):
            if index not in self.smartcontract_functions_index_position:
                self.smartcontract_functions_index_position.append(index)

        self.smartcontract_functions_index_position = sorted(
            self.smartcontract_functions_index_position
        )

        i = 0
        while i < len(self.smartcontract_functions_index_position) and i + 1 < len(
            self.smartcontract_functions_index_position
        ):
            self.smartcontract_functions_index_range.append(
                [
                    self.smartcontract_functions_index_position[i],
                    self.smartcontract_functions_index_position[i + 1] - 1,
                ]
            )
            i += 1
        print(self.smartcontract_functions_index_position)
        print("...")
        print(
            f"self.smartcontract_functions_index_range: {self.smartcontract_functions_index_range}"
        )


        if (
            len(self.smartcontract_functions_index_range) == 1
            and self.real_bytecode[-1] == "RETURN"
        ):
            self.smartcontract_functions_index_range[0][0] = (
                self.smartcontract_functions_index_range[0][0] + 2
            )
            print(
                f"final self.smartcontract_functions_index_range: {self.smartcontract_functions_index_range}"
            )

class SymbolicBytecodeExecutor(BytecodeExecutor):
    def __init__(self, symbolic_bytecode, real_bytecode):
        super().__init__(symbolic_bytecode, real_bytecode)
        self.handlers = OpcodeHandlers(self, self.generator)