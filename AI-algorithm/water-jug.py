from collections import deque

def bfs_water_jug(capA, capB, target):
  queue = deque([0, 0])  #intial to 0,0
  visited = set()       # mark visited

  while queue:
    a, b = queue.popleft()
    
    if (a, b) in visited:
      continue
      
    visited.add((a, b))   #add if not visited
    print(a, b)

    if a == target or b == target:
      print("goal reached")
      return True

    #generate next states
    next_states = [(capA, b), (a, capB), (0, b), (a, 0)]

    #pour a -> b
    transfer = min(a, capB-b)
    next_states.append(a-transfer, b+transfer)

    #pour b -> a
    transfer = min(capA-a, b)
    next_states.append(a+transfer, b-transfer)

    for state in next_states:
      if state not in visited:
        queue.append(state)

  print("no solution!")
  return False
print(bfs_water_jug(5, 3, 4))
