from math import hypot
import numpy as np

class Node:
    def __init__(self, node=None, left=None, right=None):
        self.node = node
        self.left = left
        self.right = right

class KDTree:
    def __init__(self, root):
        self.root   = Node(root)
        self.length = 0
        
    def addNode(self, new):
        axis = 0
        cur = self.root
        if not isinstance(new, Node): new = Node(new)
        
        while True:
            if new[axis] < cur[axis]:
                if cur.left is None:
                    cur.left = new
                    break
                cur = cur.left
                
            else:
                if cur.right is None:
                    cur.right = new
                    break
                cur = cur.right
            
            axis = 1 - axis
        self.length += 1
    
    def nearestNode(self, new, return_node=False):
        new = Node(new)
        node, dist = self._nearestNode(new, self.root, 0, None, float('inf'))
        return node if return_node else tuple(node.node), dist

    def _nearestNode(self, new, cur, axis, minNode, minDist):
        dist = hypot(new[0]-cur[0], new[1]-cur[1])
        if dist < minDist:
            minNode, minDist = cur, dist

        if new[axis] < cur[axis]:
            if cur.left is not None:
                minNode, minDist = self._nearestNode(new, cur.left, 1-axis, minNode, minDist)
            
            if cur.right is not None and cur[axis] - new[axis] < minDist:
                minNode, minDist = self._nearestNode(new, cur.right, 1-axis, minNode, minDist)
            
        else:
            if cur.right is not None:
                minNode, minDist = self._nearestNode(new, cur.right, 1-axis, minNode, minDist)

            if cur.left is not None and new[axis] - cur[axis] < minDist:
                minNode, minDist = self._nearestNode(new, cur.left, 1-axis, minNode, minDist)
        
        return minNode, minDist

    
class PathNode:
    def __init__(self, coords=None, parent=None):
        self.coords   = coords
        self.children = []
        self.parent   = parent

class PathTree:
    """ Tree class for generating final path """
    def __init__(self, root):
        self.root   = PathNode(root)
        self.dict   = {root: self.root}
        self.length = 0

    def addPath(self, start, end):
        newNode   = PathNode(coords=end, parent=self.dict[start])
        self.dict[start].addChild(newNode)
        self.dict[end]  = newNode


def runRRT(obstacles, start, goal, step_size, max_iter):
    circ_rad  = min(step_size/5, 5)
    final_pos = np.array(goal[:2])

    span_y = obstacles.shape[0]
    span_x = obstacles.shape[1]

    def gen_valid_rand(valid_function):
        

    KD    = KDTree(start)
    RRT   = PathTree(start)

    obstacles = Obstacles(obstacles.to_polygons())

    trials = 0
    while KD.length < max_size:
        trials += 1
        circ1.remove()

        # Select a random point q_rand \in Q_free
        q_rand = gen_valid_rand(obstacles.point_is_valid) if np.random.randint(0, 100)>5 else final_pos
        circ1 = plotter.draw_circle(q_rand, 5, time=0.01, zorder=5)
            
        # Find the nearest node and distance to it
        q_near, dist = KD.nearestNode(q_rand)
        
        # Generate the next node in the direction of q_rand
        if dist < step_size:
            if trials < 10: continue # Prevents step_size too big bug
            q_next = tuple(q_rand)
        else:
            q_next = gen_next(q_near, q_rand, step_size)
            if not obstacles.point_is_valid(*q_next): continue
        
        # Check validity and update tree
        if obstacles.check_collisions((q_near, q_next)): continue

        KD.addNode(q_next)
        RRT.addPath(q_near, q_next)

        plotter.draw_line(q_near, q_next, color='k', zorder=1, update=False)
        plotter.draw_circle(q_next, circ_rad, edgecolor='k', facecolor='w', zorder=2)

        if not obstacles.check_collisions((q_next, goal)):
            # IF there is a direct line to the goal, then TAKE IT
            goal_distance = math.hypot(q_next[0]-goal[0], q_next[1]-goal[1])
            while goal_distance > 0:
                q_new = gen_next(q_next, goal, min(goal_distance, step_size))
                RRT.addPath(q_next, q_new)
                plotter.draw_line(q_next, q_new, color='k', zorder=1, update=False)
                plotter.draw_circle(q_new, circ_rad, edgecolor='k', facecolor='w', zorder=2)
                q_next = q_new
                goal_distance -= step_size
            break

        trials = 0

    print("n =", KD.length)

    cur = RRT[goal]
    while cur.parent:
        plotter.draw_line(cur, cur.parent, update=False, color='b', zorder=3)
        plotter.draw_circle(cur, circ_rad*1.5, update=False, facecolor='xkcd:green', edgecolor='k', zorder=4)
        cur = cur.parent
    plotter.update()




