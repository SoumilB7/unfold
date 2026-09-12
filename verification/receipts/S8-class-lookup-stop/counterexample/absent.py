from torch import nn

class ClassLookupFixture(nn.Module):
    if False:
        optional_scale = 2.0

    def __init__(self, config):
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def forward(self, value):
        result = self.projection(value)
        if getattr(self, "optional_scale", None):
            result = result * self.optional_scale
        return result
