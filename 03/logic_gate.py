import numpy as np

class LogicGate:
    def __init__(self):
        pass
    
    # AND Gate
    def and_gate(self, x1, x2):
        b = -0.7
        w = np.array([0.5, 0.5, 1]) 
        x = np.array([x1, x2, b])
        y = np.sum(x * w)
        
        if y > 0:
            return 1
        else:
            return 0
    
    # OR Gate
    def or_gate(self, x1, x2):
        b = -0.2
        w = np.array([0.5, 0.5, 1])
        x = np.array([x1, x2, b])
        y = np.sum(x * w)
        
        if y > 0:
            return 1
        else:
            return 0
    
    # NAND Gate
    def nand_gate(self, x1, x2):
        b = 0.7
        w = np.array([-0.5, -0.5, 1])
        x = np.array([x1, x2, b])
        y = np.sum(x * w)
        
        if y > 0:
            return 1
        else:
            return 0
    
    # NOR Gate
    def nor_gate(self, x1, x2):
        b = -0.9
        w = np.array([1, 1, 1])
        x = np.array([x1, x2, b])
        y = np.sum(x * w)
        
        if y < 0:
            return 1
        else:
            return 0
    
    # XOR Gate (combining AND, OR, and NAND)
    def xor_gate(self, x1, x2):
        nand = self.nand_gate(x1, x2)
        or_ = self.or_gate(x1, x2)
        return self.and_gate(nand, or_)
    

if __name__ == "__main__":
    print("This class is LogicGates")
    print("It has functions for all the logic gates executed using numpy")
    print("if you want to use any logic gate type LogicGate.name_gate(input arg1, input arg2)")
    print("for example for and gate write LogicGate.and_gate(1,0)")