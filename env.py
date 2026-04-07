import random

class EnergyEnv:
    def __init__(self):
        self.max_battery = 10

    def reset(self):
        self.demand = [random.randint(3, 8) for _ in range(3)]
        self.generation = random.randint(5, 15)
        self.battery = 5
        return self._get_state()

    def _get_state(self):
        return self.demand + [self.generation, self.battery]

    def step(self, action):
        # actions: 0=equal, 1=A priority, 2=B priority, 3=C priority

        total_supply = self.generation + self.battery

        if action == 0:
            dist = [total_supply//3]*3
        elif action == 1:
            dist = [total_supply, 0, 0]
        elif action == 2:
            dist = [0, total_supply, 0]
        else:
            dist = [0, 0, total_supply]

        shortage = sum(max(0, d - s) for d, s in zip(self.demand, dist))
        wastage = max(0, total_supply - sum(dist))

        reward = - (2 * shortage + 0.5 * wastage)

        done = False
        return self._get_state(), reward, done