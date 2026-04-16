import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanograd.engine import Value
from nanograd.nn import MLP, Neuron


class TestNN:
    def test_neuron_output_is_value(self):
        n = Neuron(3)
        out = n([1.0, 2.0, 3.0])
        assert isinstance(out, Value)

    def test_neuron_parameter_count(self):
        n = Neuron(4)
        # 4 weights + 1 bias
        assert len(n.parameters()) == 5

    def test_mlp_forward_output_shape(self):
        model = MLP(3, [4, 4, 1])
        out = model([1.0, 2.0, 3.0])
        assert isinstance(out, Value)

    def test_mlp_parameters_non_empty(self):
        model = MLP(2, [3, 1])
        params = model.parameters()
        assert len(params) > 0
        assert all(isinstance(p, Value) for p in params)

    def test_mlp_backward_populates_gradients(self):
        model = MLP(2, [3, 1])
        out = model([1.0, 2.0])
        out.backward()

        params = model.parameters()
        assert any(p.grad != 0.0 for p in params)

    def test_zero_grad_resets_all_parameter_grads(self):
        model = MLP(2, [3, 1])
        out = model([0.5, 1.0])
        out.backward()

        model.zero_grad()
        assert all(p.grad == 0.0 for p in model.parameters())
