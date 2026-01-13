from typing import NewType, Optional, TypeVar

ProductType = NewType('ProductType', str)
LocationType = NewType('LocationType', str)

HierarchyType = TypeVar('HierarchyType', ProductType, LocationType)


class HierarchyTree[HierarchyType]:
    """
    A generic tree structure for representing hierarchical relationships.

    This class implements a tree node that can represent any hierarchical structure,
    where each node has a name, an optional parent, and zero or more children.

    Parameters
    ----------
    name : HierarchyType
        The name or identifier for this node in the hierarchy.
    parent : Optional[HierarchyTree[HierarchyType]], optional
        The parent node of this node in the hierarchy, by default None.
    children : Optional[list[HierarchyTree[HierarchyType]]], optional
        A list of child nodes, by default None (which creates an empty list).

    Attributes
    ----------
    name : HierarchyType
        The name or identifier for this node.
    parent : Optional[HierarchyTree[HierarchyType]]
        Reference to the parent node, or None if this is a root node.
    children : list[HierarchyTree[HierarchyType]]
        List of child nodes belonging to this node.

    Methods
    -------
    set_parent(parent)
        Set the parent node for this node.
    add_child(child)
        Add a child node to this node and set this node as the child's parent.
    is_root()
        Check if this node is a root node (has no parent).
    is_leaf()
        Check if this node is a leaf node (has no children).

    Examples
    --------
    >>> root = HierarchyTree("root")
    >>> child1 = HierarchyTree("child1")
    >>> root.add_child(child1)
    >>> root.is_root()
    True
    >>> child1.is_leaf()
    True
    """

    def __init__(
        self,
        name: HierarchyType,
        parent: Optional[HierarchyTree[HierarchyType]] = None,
        children: Optional[list[HierarchyTree[HierarchyType]]] = None,
    ):
        self.name: HierarchyType = name
        self.parent: Optional[HierarchyTree[HierarchyType]] = parent
        self.children: list[HierarchyTree[HierarchyType]] = (
            children if children is not None else []
        )
        return

    def set_parent(self, parent: HierarchyTree[HierarchyType]) -> None:
        self.parent = parent
        return

    def add_child(self, child: HierarchyTree[HierarchyType]) -> None:
        child.parent = self
        self.children.append(child)
        return

    def is_root(self) -> bool:
        return self.parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0


def add_parent(
    parent: HierarchyTree[HierarchyType],
    children: list[HierarchyTree[HierarchyType]],
) -> None:
    """
    Add a parent node to multiple children nodes.

    This function establishes parent-child relationships by adding each child
    to the parent's children list and setting the parent reference for each child.

    Parameters
    ----------
    parent : HierarchyTree[HierarchyType]
        The parent node to be assigned to all children.
    children : list[HierarchyTree[HierarchyType]]
        A list of child nodes that will be added to the parent.

    Returns
    -------
    None

    Examples
    --------
    >>> root = HierarchyTree("root")
    >>> child1 = HierarchyTree("child1")
    >>> child2 = HierarchyTree("child2")
    >>> add_parent(root, [child1, child2])
    >>> len(root.children)
    2
    """
    for child in children:
        parent.add_child(child)
        child.set_parent(parent)
    return
