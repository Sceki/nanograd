import math
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanograd.engine import Value


def _as_value(x):
    return x if isinstance(x, Value) else Value(x)


def _as_real(x):
    if isinstance(x, complex):
        return x.real
    return x


def eml_exp(x):
    return Value.eml(_as_value(x), 1.0)


def eml_log(x):
    one = Value(1.0)
    return Value.eml(one, Value.eml(Value.eml(one, _as_value(x)), one))


def eml_zero():
    return eml_log(1.0)


def eml_sub(a, b):
    return Value.eml(eml_log(_as_value(a)), eml_exp(_as_value(b)))


def eml_neg(x):
    return eml_sub(eml_zero(), _as_value(x))


def eml_add(a, b):
    return eml_sub(_as_value(a), eml_neg(_as_value(b)))


def eml_inv(x):
    return eml_exp(eml_neg(eml_log(_as_value(x))))


def eml_mul(a, b):
    return eml_exp(eml_add(eml_log(_as_value(a)), eml_log(_as_value(b))))


def eml_div(a, b):
    return eml_mul(_as_value(a), eml_inv(_as_value(b)))


def eml_tanh(x):
    ex = eml_exp(_as_value(x))
    exp_2x = eml_mul(ex, ex)
    denom = eml_add(exp_2x, 1.0)
    frac = eml_mul(2.0, eml_inv(denom))
    return eml_sub(1.0, frac)


def eml_cos(x):
    i = 1j
    exp_ix = eml_exp(eml_mul(i, _as_value(x)))
    exp_minus_ix = eml_exp(eml_mul(-i, _as_value(x)))
    return eml_div(eml_add(exp_ix, exp_minus_ix), 2.0)


def eml_sin(x):
    i = 1j
    exp_ix = eml_exp(eml_mul(i, _as_value(x)))
    exp_minus_ix = eml_exp(eml_mul(-i, _as_value(x)))
    return eml_div(eml_sub(exp_ix, exp_minus_ix), 2j)


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
        f = Value.eml(x, y)
        f.backward()
        analytical_grad_x = x.grad
        
        # Numerical gradient
        x_num = Value(x_val)
        y_num = Value(y_val)
        numerical_grad_x = numerical_gradient(
            lambda x: Value.eml(x, y_num),
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
        f = Value.eml(x, y)
        f.backward()
        analytical_grad_y = y.grad
        
        # Numerical gradient
        x_num = Value(x_val)
        y_num = Value(y_val)
        numerical_grad_y = numerical_gradient(
            lambda y: Value.eml(x_num, y),
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
            f = Value.eml(x, y)
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
        f = Value.eml(x, y)
        
        expected = math.exp(x_val) - math.log(y_val)
        assert abs(f.data - expected) < 1e-6, \
            f"Forward pass incorrect: {f.data} != {expected}"
    
    def test_gradient_accumulation(self):
        """Test that gradients accumulate correctly through backward()."""
        x = Value(1.0)
        y = Value(2.0)
        
        # First computation
        f1 = Value.eml(x, y)
        f1.backward()
        grad_x_first = x.grad
        grad_y_first = y.grad
        
        # Reset gradients
        x.grad = 0.0
        y.grad = 0.0
        
        # Second computation
        f2 = Value.eml(x, y)
        f2.backward()
        grad_x_second = x.grad
        grad_y_second = y.grad
        
        # Gradients should be the same for same inputs
        assert grad_x_first == grad_x_second
        assert grad_y_first == grad_y_second

    def test_eml_arithmetic_forward(self):
        x = 1.2
        y = 0.3

        sub_v = eml_sub(x, y)
        add_v = eml_add(x, y)
        div_v = eml_div(x, y)
        mul_v = eml_mul(x, y)

        assert abs(_as_real(sub_v.data) - (x - y)) < 1e-6
        assert abs(_as_real(add_v.data) - (x + y)) < 1e-6
        assert abs(_as_real(div_v.data) - (x / y)) < 1e-6
        assert abs(_as_real(mul_v.data) - (x * y)) < 1e-6

    def test_eml_arithmetic_gradients_wrt_x(self):
        x0 = 1.2
        y0 = 0.3

        x = Value(x0)
        y = Value(y0)
        out = eml_sub(x, y)
        out.backward()
        assert abs(_as_real(x.grad) - 1.0) < 1e-6

        x = Value(x0)
        y = Value(y0)
        out = eml_add(x, y)
        out.backward()
        assert abs(_as_real(x.grad) - 1.0) < 1e-6

        x = Value(x0)
        y = Value(y0)
        out = eml_div(x, y)
        out.backward()
        assert abs(_as_real(x.grad) - (1.0 / y0)) < 1e-6

        x = Value(x0)
        y = Value(y0)
        out = eml_mul(x, y)
        out.backward()
        assert abs(_as_real(x.grad) - y0) < 1e-6

    def test_eml_sin_cos_tanh_forward(self):
        x = 0.4
        sin_v = eml_sin(x)
        cos_v = eml_cos(x)
        tanh_v = eml_tanh(x)

        assert abs(_as_real(sin_v.data) - math.sin(x)) < 1e-6
        assert abs(_as_real(cos_v.data) - math.cos(x)) < 1e-6
        assert abs(_as_real(tanh_v.data) - math.tanh(x)) < 1e-6


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
