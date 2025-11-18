import numpy as np
from pyglm import glm


class HalfEdge:

    def __init__(self, head: 'Vertex', face: 'Face', twin: 'HalfEdge'):
        self.head = head  # the vertex at the "head" of this half-edge
        self.face = face  # left face that this half-edge borders
        self.twin = twin  # the twin half-edge (None if boundary)
        self.next = None  # the next half-edge in the face (to be set later)
        self.edge_collapse_data = None  # data for edge collapse operation, to be set later
        if head.he is None:
            head.he = self  # set the vertex's half-edge if not already set
        if face.he is None:
            face.he = self  # set the face's half-edge if not already set

    def tail(self):
        """ Get the tail of this half-edge."""
        he = self
        while he.next is not self:
            he = he.next  # previous half edge
        return he.head

    def __str__(self):
        return f"~~ HE with Head {self.head.index}, Face {self.face.index} ~~"


class Face:

    def __init__(self, index: int, he: HalfEdge = None):
        """ A face in the half-edge data structure. """
        self.index = index
        self.he = he  # one of the half-edges bordering this face
        self.normal = None  # normal of this face, for visualization and otherwise only for inital quadric computation
        self.center = None  # center of this face, for visualization
        self.M = None  # model matrix for text rendering
        self.text_scale = None  # scale for text rendering

    def get_normal(self):
        """ Return this face's normal. Will compute when called for the first time on a face."""
        if self.normal is not None:
            return self.normal
        v0 = self.he.head.pos
        v1 = self.he.next.head.pos
        v2 = self.he.next.next.head.pos
        n = glm.normalize(glm.cross(glm.vec3(*(v1 - v0)), glm.vec3(*(v2 - v0))))
        self.normal = n
        return n

    def get_center(self):
        """ Return this face's centroid. Will compute when called for the first time on a face."""
        if self.center is not None:
            return self.center
        v0 = self.he.head.pos
        v1 = self.he.next.head.pos
        v2 = self.he.next.next.head.pos
        c = (v0 + v1 + v2) / 3.0
        self.center = c
        return c

    def draw_debug(self, P: glm.mat4, V: glm.mat4, faces: np.ndarray, vert_objs: list, text_renderer):
        """Render the index of this face, for debug purposes."""
        if self.M is None:
            # Cache the necessary quantities, with redo/undo of collapses causing cache recompute by setting M to None
            # We're using the np faces array to get the vertex indices for this face
            #   because the half-edge structure may have changed
            v0 = vert_objs[faces[self.index, 0]].pos
            v1 = vert_objs[faces[self.index, 1]].pos
            v2 = vert_objs[faces[self.index, 2]].pos
            ave_edge_length = (glm.length(v0 - v1) + glm.length(v1 - v2) + glm.length(v2 - v0)) / 3.0
            center = (v0 + v1 + v2) / 3.0
            n = glm.cross(v1 - v0, v2 - v0)
            if glm.length(n) < 1e-6:
                n = glm.vec3(0, 0, 1)
            else:
                n = glm.normalize(n)
            t = glm.normalize(v1 - v0)
            b = glm.normalize(glm.cross(n, t))
            self.M = glm.mat4(
                glm.vec4(t, 0.0),  # X axis
                glm.vec4(b, 0.0),  # Y axis
                glm.vec4(n, 0.0),  # Z axis
                glm.vec4(center + n * 0.01, 1.0)  # Translation
            )
            self.text_scale = ave_edge_length * 0.1
        text_renderer.render_text(str(self.index), P, V * self.M, color=glm.vec4(1, 1, 1, 1),
                                  char_width=self.text_scale, centered=True, view_aligned=False)

    def __str__(self):
        return f"~~ Face with Index {self.index}, Referencing HE {self.he} ~~"


class Vertex:

    def __init__(self, index: int, pos: np.ndarray, he: HalfEdge):
        """ A vertex in the half-edge data structure
        Args:
            index: index of this vertex in the vertex list
            pos: 3D position of this vertex (np for convenience, as this is coming from trimesh)
            he: one of the half-edges ending at this vertex
        """
        self.index = index
        self.pos = glm.vec3(*pos)  # 3D position of this vertex
        self.Q = glm.mat4(1)  # Quadric
        self.he = he  # one of the half-edges ending at this vertex
        self.normal = None  # average normal of faces around this vertex, for visualization
        self.removed_at_level = None  # level of detail at which this vertex was removed
        self.cost = 0  # cost of this vertex living where it is

        self.text_pos = None  # Data for debug text
        self.text_scale = None

    def compute_Q(self):
        """ Compute the quadric for this vertex from the surrounding faces.
        It gets stored in the parameter self.Q"""

        self.Q = glm.mat4(0)

        # TODO: Objective 5: Compute the quadric matrix Q for this vertex
        #walk around the vertex and accumulate quadric from each adjacent face
        h = self.he
        if h is None:
            return

        while True:
            #get the face normal (unit vector)
            face_normal = h.face.get_normal()

            #compute the plane equation: Ax + By + Cz + D = 0
            A = face_normal.x
            B = face_normal.y
            C = face_normal.z
            D = -(A * self.pos.x + B * self.pos.y + C * self.pos.z)

            #let p = [A, B, C, D], Compute Kp = p * p^T
            Kp = glm.mat4(
                A * A, A * B, A * C, A * D,
                A * B, B * B, B * C, B * D,
                A * C, B * C, C * C, C * D,
                A * D, B * D, C * D, D * D
            )

            # Accumulate to Q
            self.Q += Kp

            # Move to next face around this vertex
            h = h.next.twin
            if h == self.he:
                break

    def get_normal(self) -> glm.vec3:
        """ Compute the average normal of faces adjacent to this vertex.
        The value is cached after first computation.
        This is currently only used for visualization, but could also be
        used for smooth shading of the mesh."""
        if self.normal is not None:
            return self.normal
        n = glm.vec3(0, 0, 0)
        h = self.he
        while True:
            # Accumulate normal value
            n += h.face.get_normal()
            h = h.next.twin
            if h == self.he:
                break
        if glm.length(n) > 1e-6:
            n = glm.normalize(n)
        self.normal = n
        return n

    def compute_debug_viz_data(self):
        """ Compute data for debug visualization (text position and scale)
        Note that this should be called when the vertex is first created so
        that it has access to a valid half-edge structure around it."""
        # Use the average edge length around this vertex to scale the text
        edge_length = 0.0
        num_edges = 0
        h = self.he
        while True:
            tail = h.tail().pos
            edge_length += glm.length(self.pos - tail)
            num_edges += 1
            h = h.next.twin
            if h == self.he:
                break
        avg_edge_length = edge_length / num_edges if num_edges > 0 else 0.0
        self.text_scale = avg_edge_length * 0.1
        self.text_pos = self.pos + self.get_normal() * avg_edge_length * 0.1

    def draw_debug(self, P: glm.mat4, V: glm.mat4, text_renderer):
        """Render the index of this vertex, for debug purposes."""
        text_renderer.render_text(str(self.index), P, V, pos=self.text_pos, char_width=self.text_scale,
                                  color=glm.vec4(0, 0.75, 0, 1), centered=True, view_aligned=True)

    def __str__(self):
        return f"~~ Vertex with Index {self.index}, Referencing HE {self.he} ~~"


class EdgeCollapseData:
    """ Data structure to hold the data for an edge collapse operation, comparable by cost (i.e., for priority queue)"""

    def __init__(self, he: HalfEdge):
        """ Compute the edge collapse data for the given half-edge.
        Store the cost, optimal position, and quadric matrix for the edge collapse. """

        self.he = he
        # store link to Edge collapse data in both half edges
        self.he.edge_collapse_data = self
        self.he.twin.edge_collapse_data = self

        # TODO: Objective 5: Compute cost, optimal position, and quadric matrix for edge collapse
        # TODO: change the following dummy values!
        #get the head and tail
        v_tail = he.tail()
        v_head = he.head

        # Sum their quadrics
        self.Q = v_tail.Q + v_head.Q

        A = glm.mat3(
            self.Q[0][0], self.Q[0][1], self.Q[0][2],
            self.Q[1][0], self.Q[1][1], self.Q[1][2],
            self.Q[2][0], self.Q[2][1], self.Q[2][2]
        )

        b = glm.vec3(self.Q[3][0], self.Q[3][1], self.Q[3][2])

        #optimal position is v_opt = -A^(-1) * b
        det_A = glm.determinant(A)

        #set a small regularization parameter
        lambda_reg = 1e-6

        if abs(det_A) != 0:
            # A is invertible
            try:
                A_inv = glm.inverse(A)
                self.v_opt = -A_inv * b
            except:
                # Fallback to midpoint if inverse fails
                self.v_opt = (v_tail.pos + v_head.pos) * 0.5
        else:
            # A is rank-deficient, use regularization and add regularization: λ(v - q)^T(v - q) where q is the midpoint
            q = (v_tail.pos + v_head.pos) * 0.5

            Q_reg = glm.mat4(
                1.0, 0.0, 0.0, -q.x,
                0.0, 1.0, 0.0, -q.y,
                0.0, 0.0, 1.0, -q.z,
                -q.x, -q.y, -q.z, glm.dot(q, q)
            )

            # Add regularization to Q
            Q_regularized = self.Q + lambda_reg * Q_reg

            A = glm.mat3(
                Q_regularized[0][0], Q_regularized[0][1], Q_regularized[0][2],
                Q_regularized[1][0], Q_regularized[1][1], Q_regularized[1][2],
                Q_regularized[2][0], Q_regularized[2][1], Q_regularized[2][2]
            )
            b = glm.vec3(Q_regularized[3][0], Q_regularized[3][1], Q_regularized[3][2])

            try:
                A_inv = glm.inverse(A)
                self.v_opt = -A_inv * b
            except:
                # If still fails, use midpoint
                self.v_opt = q

        # Compute the cost v_opt^T * Q * v_opt
        # Convert v_opt to homogeneous coordinates
        v_opt_homogeneous = glm.vec4(self.v_opt.x, self.v_opt.y, self.v_opt.z, 1.0)
        temp = self.Q * v_opt_homogeneous
        self.cost = glm.dot(v_opt_homogeneous, temp)

    def __lt__(self, other):
        if self.cost == other.cost:
            return id(self) < id(other)  # ensure a consistent ordering
        return self.cost < other.cost

    def __eq__(self, other):
        return id(self) == id(other)  # equal cost is not enough, must be the same edge


class CollapseRecord:
    """ data structure to hold the data for an edge collapse operation, for LOD tracking.
        Use a list of Faces, rather than indices, as face indices will change as we collapse."""

    def __init__(self, affected_faces: list[Face], old_indices: np.ndarray, new_indices: np.ndarray):
        self.affected_faces = affected_faces  # faces that were removed during this collapse
        self.old_indices = old_indices.copy()  # to be safe, make our own copy
        self.new_indices = new_indices.copy()

    def redo(self, faces: np.ndarray):
        """ Apply this collapse record to the given faces array."""
        for i, f in enumerate(self.affected_faces):
            f.M = None  # invalidate cached model matrix for text rendering
            faces[f.index, :] = self.new_indices[i]

    def undo(self, faces: np.ndarray):
        """ Undo this collapse record on the given faces array. """
        for i, f in enumerate(self.affected_faces):
            f.M = None  # invalidate cached model matrix for text rendering
            faces[f.index, :] = self.old_indices[i]


def build_heds(F: np.ndarray, vert_objs: list[Vertex]) -> (list[HalfEdge], list[Face]):
    """ Build a half-edge data structure from the given vertices and faces.
    Args:
        F: (num_faces, 3) array of vertex indices for each triangular face
        vert_objs: a list of vertices to set as head of the half edges
    Returns:
        List of *all* HalfEdge objects
        List of *all* Face objects
    """

    he_list = []
    face_objs = []
    
    # TODO: Objective 1: Build the half-edge data structure
    #Keep track of halfedges
    edge_dict = {}
    for face_idex in range(F.shape[0]):
        v0, v1, v2 = F[face_idex]

        face = Face(face_idex)
        face_objs.append(face)

        #create halfedges
        he0 = HalfEdge(vert_objs[v1], face, None)
        he1 = HalfEdge(vert_objs[v2], face, None)
        he2 = HalfEdge(vert_objs[v0], face, None)

        he0.next = he1
        he1.next = he2
        he2.next = he0

        he_list.extend([he0, he1, he2])

        #store half edges, format: [(tail, head)]
        edge_dict[(v0, v1)] = he0
        edge_dict[(v1, v2)] = he1
        edge_dict[(v2, v0)] = he2

        #add twins
        for he in he_list:
            if he.twin is None:
                tail = he.tail().index
                head = he.head.index

                twin_key = (head, tail)

                if twin_key in edge_dict:
                    twin = edge_dict[twin_key]
                    he.twin = twin
                    twin.twin = he

    return he_list, face_objs