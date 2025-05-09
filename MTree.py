import heapq
import bisect
from utility import * 

class MTree:
    def __init__(self, node_size=4, KNN_size=4, distance_metric = 'euclidean'):
        self.node_size = node_size
        self.KNN_size = KNN_size
        self.root = MTree.Node(node_size=node_size)
        self.distance_metric = distance_metric


    class Node:
        def __init__(self, node_size=4, is_leaf=True, parent=None):
            self.entries = []
            self.is_leaf = is_leaf
            self.parent = parent
            self.radius = 0
            self.node_size = node_size

        def is_full(self):
            return len(self.entries) >= self.node_size
        def __lt__(self, other):
            return id(self) < id(other)


    class Entry:
        def __init__(self, obj, radius=0, subtree=None):
            self.obj = obj
            self.radius = radius
            self.subtree = subtree  # Only for non-leaf nodes

    def KNN_search(self, query, k=None):
        if k is None:
            k = self.KNN_size
        neighbours = []
        pq = []
        heapq.heappush(pq, (0, self.root))
        d_k = float('inf')
        
        while pq:
            dist, node = heapq.heappop(pq)
            if dist > d_k:
                continue
            d_k = self.KNN_node_search(node, query, k, d_k, neighbours)
            
        return [x[1] for x in neighbours]

    def KNN_node_search(self, node, query, k, d_k, neighbours):
        pq = []
        if not node.is_leaf:
            for entry in node.entries:
                dist = compute_distance(entry.obj, query, self.distance_metric)
                min_dist = max(0, dist - entry.radius)
                if min_dist <= d_k:
                    heapq.heappush(pq, (min_dist, entry.subtree))
        else:
            for entry in node.entries:
                dist = compute_distance(entry.obj, query, self.distance_metric)
                if dist <= d_k:
                    d_k = self.NN_update( (dist, entry.obj), k, neighbours)
        
        # Process non-leaf nodes
        while pq and pq[0][0] <= d_k:
            dist, child = heapq.heappop(pq)
            d_k = self.KNN_node_search(child, query, k, d_k, neighbours)
            
        return d_k

    def NN_update(self, new_entry, k, neighbours):
        distance, obj = new_entry
        insert_pos = bisect.bisect_left([x[0] for x in neighbours], distance)
        neighbours.insert(insert_pos, (distance, obj))
        
        if len(neighbours) > k:
            neighbours = neighbours[:k]
        
        return neighbours[-1][0] if len(neighbours) >= k else float('inf')

    def insert(self, obj, node=None):
        if node is None:
            node = self.root
            
        if not node.is_leaf:
            # Find the best entry to insert into
            best_entry = None
            min_dist = float('inf')
            min_enlargement = float('inf')
            
            for entry in node.entries:
                dist = compute_distance(entry.obj, obj, self.distance_metric)
                if dist <= entry.radius:
                    enlargement = dist - entry.radius
                    if enlargement < min_enlargement or (enlargement == min_enlargement and dist < min_dist):
                        min_enlargement = enlargement
                        min_dist = dist
                        best_entry = entry
                        
            if best_entry:
                self.insert(obj, best_entry.subtree)
                # Update radius if needed
                dist = compute_distance(best_entry.obj, obj, self.distance_metric)
                if dist > best_entry.radius:
                    best_entry.radius = dist
                return
            else:
                # No suitable entry found, find entry requiring minimal radius enlargement
                best_entry = min(node.entries, 
                               key=lambda e: compute_distance(e.obj, obj, self.distance_metric) - e.radius)
                self.insert(obj, best_entry.subtree)
                # Update radius
                dist = compute_distance(best_entry.obj, obj, self.distance_metric)
                best_entry.radius = max(best_entry.radius, dist)
                return
        else:
            # Leaf node insertion
            if not node.is_full():
                new_entry = self.Entry(obj)
                node.entries.append(new_entry)
            else:
                self.split(node, obj)

    def split(self, node, new_obj):
        # Create new node
        new_node = MTree.Node(node_size=self.node_size, is_leaf=node.is_leaf, parent=node.parent)        
        # Get all objects including the new one
        all_entries = node.entries.copy()
        if node.is_leaf:
            all_entries.append(self.Entry(new_obj))
        else:
            all_entries.append(self.Entry(new_obj.obj, new_obj.radius, new_obj.subtree))
        
        # Promote two entries
        promoted1, promoted2 = self.promote(all_entries)
        
        # Partition entries
        group1, group2 = self.partition(all_entries, promoted1.obj, promoted2.obj)
        
        # Update original node
        node.entries = group1
        node.radius = max(compute_distance(promoted1.obj, e.obj, self.distance_metric) for e in group1) if group1 else 0
        
        # Update new node
        new_node.entries = group2
        new_node.radius = max(compute_distance(promoted2.obj, e.obj, self.distance_metric) for e in group2) if group2 else 0
        
        # Handle parent updates
        if node is self.root:
            # Create new root
            new_root = MTree.Node(node_size=self.node_size, is_leaf=False)
            new_root.entries = [
                self.Entry(promoted1.obj, node.radius, node),
                self.Entry(promoted2.obj, new_node.radius, new_node)
            ]
            node.parent = new_root
            new_node.parent = new_root
            self.root = new_root
        else:
            # Update parent
            parent = node.parent
            # Replace the original entry with promoted1
            for i, entry in enumerate(parent.entries):
                if entry.subtree == node:
                    parent.entries[i] = self.Entry(promoted1.obj, node.radius, node)
                    break
            
            # Add promoted2 to parent
            new_parent_entry = self.Entry(promoted2.obj, new_node.radius, new_node)
            new_node.parent = parent
            
            if parent.is_full():
                self.split(parent, new_parent_entry)
            else:
                parent.entries.append(new_parent_entry)

    def promote(self, entries):
        max_distance = -1
        best_pair = None
        
        for i in range(len(entries)):
            for j in range(i+1, len(entries)):
                dist = compute_distance(entries[i].obj, entries[j].obj, self.distance_metric)
                if dist > max_distance:
                    max_distance = dist
                    best_pair = (entries[i], entries[j])
        
        return best_pair[0], best_pair[1]

    def partition(self, entries, obj1, obj2, strategy='closest'):
        group1 = []
        group2 = []
        
        for entry in entries:
            dist1 = compute_distance(entry.obj, obj1, self.distance_metric)
            dist2 = compute_distance(entry.obj, obj2, self.distance_metric)
            
            if dist1 < dist2:
                group1.append(entry)
            else:
                group2.append(entry)
        
        return group1, group2
