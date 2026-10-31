def objective_fn(x):
  return -x**2 + 6*x

def get_neighbours(x):
  return [x-1, x+1]

def hill_climbing(intial_state):
  current = intial_state

  while True:
    neighbours = get_neighbours(current)
    neighbour = max(neighbours, key=objective_fn)
    
    if objective_fn(neighbour) <= objective_fn(current):
      return current

    current = neighbour

#driver code

start = 1
solution = hill_climbing(start)

print("Starting State:", start)
print("Best State Found:", solution)
print("Maximum Value:", objective_fn(solution))
