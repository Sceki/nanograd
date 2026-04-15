import math
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanograd.engine import Value


def numerical_gradient(f, x, eps=1e-5):
    """Compute numerical gradient using finite differences."""
    x_plus = Value(x.data + eps)
    x_minus = Value(x.data - eps)
    
    grad = (f(x_plus).data - f(x_minus).data) / (2 * eps)
    return grad


class TestGradients:
    """Test suite for gradient correctness using numerical gradient checking."""
    
    def test_eml_gradient_wrt_x(self):
        """Test that d/dx of eml(x, y) = exp(x) is correct."""
        x_val = 2.0
        y_val = 3.0
        
        # Analytical gradient
        x = Value(x_val)
        y = Value(y_val)
        f = x.eml(y)
        f.backward()
        analytical_grad_x = x.grad
        
        # Numerical gradient
        x_num = Value(x_val)
        y_num = Value(y_val)
        numerical_grad_x = numerical_gradient(
            lambda x: x.eml(y_num),
            x_num
        )
        
        # Check they match
        assert abs(analytical_grad_x - numerical_grad_x) < 1e-4, \
            f"Gradient wrt x mismatch: {analytical_grad_x} vs {numerical_grad_x}"
        assert abs(analytical_grad_x - math.exp(x_val)) < 1e-6, \
            f"Gradient should be exp({x_val}) = {math.exp(x_val)}, got {analytical_grad_x}"
    
    def test_eml_gradient_wrt_y(self):
        """Test that d/dy of eml(x, y) = -1/y is correct."""
        x_val = 2.0
        y_val = 3.0
        
        # Analytical gradient
        x = Value(x_val)
        y = Value(y_val)
        f = x.eml(y)
        f.backward()
        analytical_grad_y = y.grad
        
        # Numerical gradient
        x_num = Value(x_val)
        y_num = Value(y_val)
        numerical_grad_y = numerical_gradient(
            lambda y: x_num.eml(y),
            y_num
        )
        
        # Check they match
        assert abs(analytical_grad_y - numerical_grad_y) < 1e-4, \
            f"Gradient wrt y mismatch: {analytical_grad_y} vs {numerical_grad_y}"
        assert abs(analytical_grad_y - (-1.0 / y_val)) < 1e-6, \
            f"Gradient should be -1/{y_val} = {-1.0/y_val}, got {analytical_grad_y}"
    
    def test_eml_various_values(self):
        """Test gradients with various input values."""
        test_cases = [
            (1.0, 2.0),
            (0.5, 1.5),
            (10.0, 5.0),
            (-2.0, 1.0),
        ]
        
        for x_val, y_val in test_cases:
            # Analytical
            x = Value(x_val)
            y = Value(y_val)
            f = x.eml(y)
            f.backward()
            
            # Verify analytical gradients match expected formulas
            expected_grad_x = math.exp(x_val)
            expected_grad_y = -1.0 / y_val
            
            assert abs(x.grad - expected_grad_x) < 1e-6, \
                f"At ({x_val}, {y_val}): grad_x {x.grad} != {expected_grad_x}"
            assert abs(y.grad - expected_grad_y) < 1e-6, \
                f"At ({x_val}, {y_val}): grad_y {y.grad} != {expected_grad_y}"
    
    def test_forward_pass_eml(self):
        """Test that forward pass computes exp(x) - log(y) correctly."""
        x_val = 2.0
        y_val = 3.0
        
        x = Value(x_val)
        y = Value(y_val)
        f = x.eml(y)
        
        expected = math.exp(x_val) - math.log(y_val)
        assert abs(f.data - expected) < 1e-6, \
            f"Forward pass incorrect: {f.data} != {expected}"
    
    def test_gradient_accumulation(self):
        """Test that gradients accumulate correctly through backward()."""
        x = Value(1.0)
        y = Value(2.0)
        
        # First computation
        f1 = x.eml(y)
        f1.backward()
        grad_x_first = x.grad
        grad_y_first = y.grad
        
        # Reset gradients
        x.grad = 0.0
        y.grad = 0.0
        
        # Second computation
        f2 = x.eml(y)
        f2.backward()
        grad_x_second = x.grad
        grad_y_second = y.grad
        
        # Gradients should be the same for same inputs
        assert grad_x_first == grad_x_second
        assert grad_y_first == grad_y_second


if __name__ == '__main__':
    # Run all tests
    test_suite = TestGradients()
    
    print("Running test_forward_pass_eml...")
    test_suite.test_forward_pass_eml()
    print("✓ PASSED")
    
    print("Running test_eml_gradient_wrt_x...")
    test_suite.test_eml_gradient_wrt_x()
    print("✓ PASSED")
    
    print("Running test_eml_gradient_wrt_y...")
    test_suite.test_eml_gradient_wrt_y()
    print("✓ PASSED")
    
    print("Running test_eml_various_values...")
    test_suite.test_eml_various_values()
    print("✓ PASSED")
    
    print("Running test_gradient_accumulation...")
    test_suite.test_gradient_accumulation()
    print("✓ PASSED")
    
    print("\n✅ All tests passed!")
