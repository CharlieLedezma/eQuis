import sympy as sym
import numpy as np

class FormulaEvaluator:
    def __init__(self):
        a,b,c = sym.symbols('a,b,c')

        """         a = sym.Symbol("a")
                b = sym.Symbol("b")
                c = sym.Symbol("c") """
        self.P1 = sym.Matrix([sym.sin(a**2), b**3, c])
        self.expr =sym.lambdify((a,b,c),self.P1,'numpy')

    
    #def evaluate(self, P1):
    #    expr =sym.lambdify((a,b,c),P1,'numpy')
    #    return self._lambdified_func(value_x, value_y)

    def lamb_P1(self,a,b,c):
        return self.expr(a,b,c)
    
#Implementation 



if __name__ == "__main__":

    evaluator = FormulaEvaluator()
    single_result = evaluator.lamb_P1(1.5,3,7.8)
    print(f"Evaluation of the test: {single_result}") 
    