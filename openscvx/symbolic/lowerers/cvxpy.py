"""CVXPy backend for lowering symbolic expressions to CVXPy format.

This module implements the CVXPy lowering backend that converts symbolic expression
AST nodes into CVXPy expressions for convex optimization. The lowering uses a visitor
pattern where each expression type has a corresponding visitor method.

Architecture:
    The CVXPy lowerer follows a visitor pattern with centralized registration:

    1. **Visitor Registration**: The @visitor decorator registers handler functions
       for each expression type in the _CVXPY_VISITORS dictionary
    2. **Dispatch**: The dispatch() function looks up and calls the appropriate
       visitor based on the expression's type
    3. **Recursive Lowering**: Each visitor recursively lowers child expressions
       and composes CVXPy operations
    4. **Translation Only**: This module only translates expressions; CVXPy itself
       validates DCP (Disciplined Convex Programming) rules when the problem is
       constructed/solved

Key Features:
    - **Expression Translation**: Converts symbolic AST to CVXPy expression format
    - **Variable Management**: Maps symbolic States/Controls to CVXPy variables
      through a variable_map dictionary
    - **Parameter Support**: Handles both constant parameters and CVXPy Parameters
      for efficient parameter sweeps
    - **Constraint Generation**: Produces CVXPy constraint objects from symbolic
      equality and inequality expressions

Backend Usage:
    CVXPy lowering is used for convex constraints in the SCP subproblem. Unlike
    JAX lowering (which happens early during problem construction), CVXPy lowering
    occurs later during Problem.initialize() when CVXPy variables are
    available. See lower_symbolic_expressions() in symbolic/lower.py for details.

CVXPy Variable Mapping:
    The lowerer requires a variable_map dictionary that maps symbolic variable names
    to CVXPy expressions. For trajectory optimization::

        variable_map = {
            "x": cvxpy.Variable((n_x,)),  # State vector
            "u": cvxpy.Variable((n_u,)),  # Control vector
            "param_name": cvxpy.Parameter((3,)),  # Runtime parameters
        }

    States and Controls use their slices (assigned during unification) to extract
    the correct portion of the unified x and u vectors.

Example:
    Basic usage::

        import cvxpy as cp
        from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer
        import openscvx as ox

        # Create symbolic expression
        x = ox.State("x", shape=(3,))
        u = ox.Control("u", shape=(2,))
        expr = ox.Norm(x)**2 + 0.1 * ox.Norm(u)**2

        # Create CVXPy variables
        cvx_x = cp.Variable(3)
        cvx_u = cp.Variable(2)

        # Lower to CVXPy
        lowerer = CvxpyLowerer(variable_map={"x": cvx_x, "u": cvx_u})
        cvx_expr = lowerer.lower(expr)

        # Use in optimization problem
        prob = cp.Problem(cp.Minimize(cvx_expr), constraints=[...])
        prob.solve()

    Constraint lowering::

        # Symbolic constraint
        constraint = ox.Norm(x) <= 1.0

        # Lower to CVXPy constraint
        cvx_constraint = lowerer.lower(constraint)

        # Add to problem
        prob = cp.Problem(cp.Minimize(cost), constraints=[cvx_constraint])

For Contributors:
    **Adding Support for New Expression Types**

    To add support for a new symbolic expression type to CVXPy lowering:

    1. **Define the visitor method** in CvxpyLowerer with the @visitor decorator::

        @visitor(MyNewExpr)
        def _visit_my_new_expr(self, node: MyNewExpr) -> cp.Expression:
            # Lower child expressions recursively
            operand = self.lower(node.operand)

            # Return CVXPy expression
            return cp.my_operation(operand)

    2. **Key requirements**:
        - Use the @visitor(ExprType) decorator to register the handler
        - Method name should be _visit_<expr_name> (private, lowercase, snake_case)
        - Recursively lower all child expressions using self.lower()
        - Return a cp.Expression or cp.Constraint object
        - Use cp.* operations for CVXPy atoms

    3. **DCP considerations**:
        - This module only translates; CVXPy validates DCP rules
        - Document the mathematical properties in the docstring (convex, concave, affine)
        - For non-DCP operations, raise NotImplementedError with helpful message
        - See _visit_sin, _visit_cos, _visit_ctcs for examples

    4. **Example patterns**:
        - Unary operation: ``return cp.my_func(self.lower(node.operand))``
        - Binary operation: ``return self.lower(node.left) + self.lower(node.right)``
        - Constraints: ``return self.lower(node.lhs) <= self.lower(node.rhs)``
        - Not supported: Raise NotImplementedError with guidance

    5. **Testing**: Ensure your visitor works with:
        - Simple expressions: Direct lowering to cp.Expression
        - Constraint validation: CVXPy accepts the result
        - DCP checking: CVXPy's problem.solve() validates correctly

See Also:
    - lower_to_cvxpy(): Convenience wrapper for single expression lowering
    - JaxLowerer: Alternative backend for non-convex constraints and dynamics
    - lower_symbolic_expressions(): Main orchestrator in symbolic/lower.py
    - CVXPy documentation: https://www.cvxpy.org/
"""

from typing import Callable, Dict, Type

import cvxpy as cp

from openscvx.symbolic.expr import (
    CTCS,
    Abs,
    Add,
    All,
    Any,
    Bilerp,
    Block,
    Concat,
    Cond,
    Constant,
    Cos,
    CrossNodeConstraint,
    Div,
    Equality,
    Exp,
    Expr,
    Hstack,
    Huber,
    Index,
    Inequality,
    Linterp,
    Log,
    LogSumExp,
    MatMul,
    Max,
    Mul,
    Neg,
    NodeReference,
    Norm,
    Parameter,
    PositivePart,
    Power,
    Sin,
    SmoothReLU,
    Sqrt,
    Square,
    Stack,
    Sub,
    Sum,
    Tan,
    Transpose,
    Vstack,
)
from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.expr.linalg import Inv
from openscvx.symbolic.expr.state import State
from openscvx.symbolic.time import Time

_CVXPY_VISITORS: Dict[Type[Expr], Callable] = {}
"""Registry mapping expression types to their visitor functions."""


def visitor(expr_cls: Type[Expr]) -> Callable[[Callable], Callable]:
    """Decorator to register a visitor function for an expression type.

    This decorator registers a visitor method to handle a specific expression
    type during CVXPy lowering. The decorated function is stored in _CVXPY_VISITORS
    and will be called by dispatch() when lowering that expression type.

    Args:
        expr_cls: The Expr subclass this visitor handles (e.g., Add, Mul, Norm)

    Returns:
        Decorator function that registers the visitor and returns it unchanged

    Example:
        Register a function as the visitor for the Add expression:

            @visitor(Add)
            def _visit_add(self, node: Add):
                # Lower addition to CVXPy
                ...

    Note:
        Multiple expression types can share a visitor by stacking decorators::

            @visitor(Equality)
            @visitor(Inequality)
            def _visit_constraint(self, node: Constraint):
                # Handle both equality and inequality
                ...
    """

    def register(fn: Callable[[Any, Expr], cp.Expression]):
        _CVXPY_VISITORS[expr_cls] = fn
        return fn

    return register


def dispatch(lowerer: "CvxpyLowerer", expr: Expr) -> cp.Expression:
    """Dispatch an expression to its registered visitor function.

    Looks up the visitor function for the expression's type and calls it.
    This is the core of the visitor pattern implementation.

    Args:
        lowerer: The CvxpyLowerer instance (provides context for visitor methods)
        expr: The expression node to lower

    Returns:
        The result of calling the visitor function (CVXPy expression or constraint)

    Raises:
        NotImplementedError: If no visitor is registered for the expression type

    Example:
        Dispatch an expression to lower it:

            lowerer = CvxpyLowerer(variable_map={...})
            expr = Add(x, y)
            cvx_expr = dispatch(lowerer, expr)  # Calls visit_add
    """
    fn = _CVXPY_VISITORS.get(type(expr))
    if fn is None:
        raise NotImplementedError(
            f"{lowerer.__class__.__name__!r} has no visitor for {type(expr).__name__}"
        )
    return fn(lowerer, expr)


class CvxpyLowerer:
    """CVXPy backend for lowering symbolic expressions to disciplined convex programs.

    This class implements the visitor pattern for converting symbolic expression
    AST nodes to CVXPy expressions and constraints. Each expression type has a
    corresponding visitor method decorated with @visitor that handles the lowering
    logic.

    The lowering process is recursive: each visitor lowers its child expressions
    first, then composes them into a CVXPy operation. CVXPy will validate DCP
    (Disciplined Convex Programming) compliance when the problem is constructed.

    Attributes:
        variable_map (dict): Dictionary mapping variable names to CVXPy expressions.
            Must include "x" for states and "u" for controls. May include parameter
            names mapped to CVXPy Parameter objects or constants.

    Example:
        Lower an expression to CVXPy:

            import cvxpy as cp
            lowerer = CvxpyLowerer(variable_map={
                "x": cp.Variable(3),
                "u": cp.Variable(2),
            })
            expr = ox.Norm(x)**2 + 0.1 * ox.Norm(u)**2
            cvx_expr = lowerer.lower(expr)

    Note:
        The lowerer is stateful (stores variable_map) unlike JaxLowerer which
        is stateless. Variables must be registered before lowering expressions
        that reference them.
    """

    def __init__(self, variable_map: Dict[str, cp.Expression] = None):
        """Initialize the CVXPy lowerer.

        Args:
            variable_map: Dictionary mapping variable names to CVXPy expressions.
                For State/Control objects, keys should be "x" and "u" respectively.
                For Parameter objects, keys should match their names. If None, an
                empty dictionary is created.

        Example:
            Initialize the CVXPy lowerer with the variable map:

                cvx_x = cp.Variable(3, name="x")
                cvx_u = cp.Variable(2, name="u")
                lowerer = CvxpyLowerer({"x": cvx_x, "u": cvx_u})
        """
        self.variable_map = variable_map or {}

    def lower(self, expr: Expr) -> cp.Expression:
        """Lower a symbolic expression to a CVXPy expression.

        Main entry point for lowering. Delegates to dispatch() which looks up
        the appropriate visitor method based on the expression type.

        Args:
            expr: Symbolic expression to lower (any Expr subclass)

        Returns:
            CVXPy expression or constraint object. For arithmetic expressions,
            returns cp.Expression. For Equality/Inequality, returns cp.Constraint.

        Raises:
            NotImplementedError: If no visitor exists for the expression type
            ValueError: If required variables are not in variable_map

        Example:
            Lower an expression to a CVXPy expression:

                lowerer = CvxpyLowerer(variable_map={"x": cvx_x, "u": cvx_u})
                x = ox.State("x", shape=(3,))
                expr = ox.Norm(x)
                cvx_expr = lowerer.lower(expr)
        """
        return dispatch(self, expr)

    def register_variable(self, name: str, cvx_expr: cp.Expression):
        """Register a CVXPy variable/expression for use in lowering.

        Adds or updates a variable in the variable_map. Useful for dynamically
        adding variables after the lowerer has been created.

        Args:
            name: Variable name (e.g., "x", "u", or parameter name)
            cvx_expr: CVXPy expression to associate with the name

        Example:
            Register a variable:

                lowerer = CvxpyLowerer()
                lowerer.register_variable("x", cp.Variable(3))
                lowerer.register_variable("obs_center", cp.Parameter(3))
        """
        self.variable_map[name] = cvx_expr

    @visitor(Constant)
    def _visit_constant(self, node: Constant) -> cp.Expression:
        """Lower a constant value to a CVXPy constant.

        Wraps the constant's numpy array value in a CVXPy Constant expression.

        Args:
            node: Constant expression node

        Returns:
            CVXPy constant expression wrapping the value
        """
        return cp.Constant(node.value)

    @visitor(Time)
    @visitor(State)
    def _visit_state(self, node: State) -> cp.Expression:
        """Lower a state variable to a CVXPy expression.

        Extracts the appropriate slice from the unified state vector "x" using
        the slice assigned during unification. The "x" variable must exist in
        the variable_map.

        Args:
            node: State expression node (or Time, which is a State subclass)

        Returns:
            CVXPy expression representing the state slice: x[slice]

        Raises:
            ValueError: If "x" is not found in variable_map
        """
        if "x" not in self.variable_map:
            raise ValueError("State vector 'x' not found in variable_map.")

        cvx_var = self.variable_map["x"]

        # If the state has a slice assigned, apply it
        if node._slice is not None:
            return cvx_var[node._slice]
        return cvx_var

    @visitor(Control)
    def _visit_control(self, node: Control) -> cp.Expression:
        """Lower a control variable to a CVXPy expression.

        Extracts the appropriate slice from the unified control vector "u" using
        the slice assigned during unification. The "u" variable must exist in
        the variable_map.

        Args:
            node: Control expression node

        Returns:
            CVXPy expression representing the control slice: u[slice]

        Raises:
            ValueError: If "u" is not found in variable_map
        """
        if "u" not in self.variable_map:
            raise ValueError("Control vector 'u' not found in variable_map.")

        cvx_var = self.variable_map["u"]

        # If the control has a slice assigned, apply it
        if node._slice is not None:
            return cvx_var[node._slice]
        return cvx_var

    @visitor(NodeReference)
    def _visit_node_reference(self, node: "NodeReference") -> cp.Expression:
        """Lower NodeReference - extract value at a specific trajectory node.

        NodeReference enables cross-node constraints by referencing state/control
        values at specific discrete time points. This requires the variable_map to
        contain full trajectory arrays (N, n_x) or (N, n_u) rather than single-node
        vectors.

        Args:
            node: NodeReference expression with base and node_idx

        Returns:
            CVXPy expression representing the variable at the specified node:
            x[node_idx, slice] or u[node_idx, slice]

        Raises:
            ValueError: If the required trajectory variable is not in variable_map
            ValueError: If the base variable has no slice assigned
            NotImplementedError: If the base is a compound expression

        Example:
            For cross-node constraint: position.at(5) - position.at(4) <= 0.1

            variable_map = {
                "x": cp.vstack([x_nonscaled[k] for k in range(N)]),  # (N, n_x)
            }
            # position.at(5) lowers to x[5, position._slice]

        Note:
            The node_idx is already resolved to an absolute integer index during
            expression construction, so negative indices are already handled.
        """
        from openscvx.symbolic.expr.control import Control
        from openscvx.symbolic.expr.state import State

        idx = node.node_idx

        if isinstance(node.base, State):
            if "x" not in self.variable_map:
                raise ValueError(
                    "State vector 'x' not found in variable_map. "
                    "For cross-node constraints, 'x' must be the full trajectory (N, n_x)."
                )

            cvx_var = self.variable_map["x"]  # Should be (N, n_x) for cross-node constraints

            # Apply slice if state has one assigned
            if node.base._slice is not None:
                return cvx_var[idx, node.base._slice]
            else:
                # No slice means this is the entire unified state vector
                return cvx_var[idx, :]

        elif isinstance(node.base, Control):
            if "u" not in self.variable_map:
                raise ValueError(
                    "Control vector 'u' not found in variable_map. "
                    "For cross-node constraints, 'u' must be the full trajectory (N, n_u)."
                )

            cvx_var = self.variable_map["u"]  # Should be (N, n_u) for cross-node constraints

            # Apply slice if control has one assigned
            if node.base._slice is not None:
                return cvx_var[idx, node.base._slice]
            else:
                # No slice means this is the entire unified control vector
                return cvx_var[idx, :]

        else:
            # Compound expression (e.g., position[0].at(5))
            # This is more complex - would need to lower base in single-node context
            raise NotImplementedError(
                "Compound expressions in NodeReference are not yet supported for CVXPy lowering. "
                f"Base expression type: {type(node.base).__name__}. "
                "Only State and Control NodeReferences are currently supported."
            )

    @visitor(CrossNodeConstraint)
    def _visit_cross_node_constraint(self, node: CrossNodeConstraint) -> cp.Constraint:
        """Lower CrossNodeConstraint to CVXPy constraint.

        CrossNodeConstraint wraps constraints that reference multiple trajectory
        nodes via NodeReference (e.g., rate limits like x.at(k) - x.at(k-1) <= r).

        For CVXPy lowering, this simply lowers the inner constraint. The NodeReference
        nodes within the constraint will handle extracting values from the full
        trajectory arrays (which must be provided in variable_map as "x" and "u").

        Args:
            node: CrossNodeConstraint expression wrapping the inner constraint

        Returns:
            CVXPy constraint object

        Note:
            The variable_map must contain full trajectory arrays:
                - "x": (N, n_x) CVXPy expression (e.g., cp.vstack(x_nonscaled))
                - "u": (N, n_u) CVXPy expression (e.g., cp.vstack(u_nonscaled))

            NodeReference visitors will index into these arrays using the fixed
            node indices baked into the expression.

        Example:
            For constraint: position.at(5) - position.at(4) <= max_step

            With variable_map = {"x": cp.vstack([x[k] for k in range(N)])}

            The lowered constraint evaluates:
                x[5, pos_slice] - x[4, pos_slice] <= max_step
        """
        # Simply lower the inner constraint - NodeReference handles indexing
        return self.lower(node.constraint)

    @visitor(Parameter)
    def _visit_parameter(self, node: Parameter) -> cp.Expression:
        """Lower a parameter to a CVXPy expression.

        Parameters are looked up by name in the variable_map. They can be mapped
        to CVXPy Parameter objects (for efficient parameter sweeps) or constants.

        Args:
            node: Parameter expression node

        Returns:
            CVXPy expression from variable_map (Parameter or constant)

        Raises:
            ValueError: If parameter name is not found in variable_map

        Note:
            For parameter sweeps without recompilation, map to cp.Parameter.
            For fixed values, map to cp.Constant or numpy arrays.
        """
        param_name = node.name
        if param_name in self.variable_map:
            return self.variable_map[param_name]
        else:
            raise ValueError(
                f"Parameter '{param_name}' not found in variable_map. "
                f"Add it during CVXPy lowering or use cp.Parameter for parameter sweeps."
            )

    @visitor(Add)
    def _visit_add(self, node: Add) -> cp.Expression:
        """Lower addition to CVXPy expression.

        Recursively lowers all terms and composes them with element-wise addition.
        Addition is affine and always DCP-compliant.

        Args:
            node: Add expression node with multiple terms

        Returns:
            CVXPy expression representing the sum of all terms
        """
        terms = [self.lower(term) for term in node.terms]
        result = terms[0]
        for term in terms[1:]:
            result = result + term
        return result

    @visitor(Sub)
    def _visit_sub(self, node: Sub) -> cp.Expression:
        """Lower subtraction to CVXPy expression (element-wise left - right).

        Subtraction is affine and always DCP-compliant.

        Args:
            node: Sub expression node

        Returns:
            CVXPy expression representing left - right
        """
        left = self.lower(node.left)
        right = self.lower(node.right)
        return left - right

    @visitor(Mul)
    def _visit_mul(self, node: Mul) -> cp.Expression:
        """Lower element-wise multiplication to CVXPy expression.

        Element-wise multiplication is DCP-compliant when at least one operand
        is constant. For quadratic forms, use MatMul instead.

        Args:
            node: Mul expression node with multiple factors

        Returns:
            CVXPy expression representing element-wise product

        Note:
            For convex optimization, typically one factor should be constant.
            CVXPy will raise a DCP error if the composition violates DCP rules.
        """
        factors = [self.lower(factor) for factor in node.factors]
        result = factors[0]
        for factor in factors[1:]:
            result = result * factor
        return result

    @visitor(Div)
    def _visit_div(self, node: Div) -> cp.Expression:
        """Lower element-wise division to CVXPy expression.

        Division is DCP-compliant when the denominator is constant or when
        the numerator is constant and the denominator is concave.

        Args:
            node: Div expression node

        Returns:
            CVXPy expression representing left / right

        Note:
            CVXPy will raise a DCP error if the division violates DCP rules.
        """
        left = self.lower(node.left)
        right = self.lower(node.right)
        return left / right

    @visitor(MatMul)
    def _visit_matmul(self, node: MatMul) -> cp.Expression:
        """Lower matrix multiplication to CVXPy expression using @ operator.

        Matrix multiplication is DCP-compliant when at least one operand is
        constant. Used for quadratic forms like x.T @ Q @ x.

        Args:
            node: MatMul expression node

        Returns:
            CVXPy expression representing left @ right
        """
        left = self.lower(node.left)
        right = self.lower(node.right)
        return left @ right

    @visitor(Neg)
    def _visit_neg(self, node: Neg) -> cp.Expression:
        """Lower negation (unary minus) to CVXPy expression.

        Negation preserves DCP properties (negating convex gives concave).

        Args:
            node: Neg expression node

        Returns:
            CVXPy expression representing -operand
        """
        operand = self.lower(node.operand)
        return -operand

    @visitor(Sum)
    def _visit_sum(self, node: Sum) -> cp.Expression:
        """Lower sum reduction to CVXPy expression (sums all elements).

        Sum preserves DCP properties (sum of convex is convex).

        Args:
            node: Sum expression node

        Returns:
            CVXPy scalar expression representing the sum of all elements
        """
        operand = self.lower(node.operand)
        return cp.sum(operand)

    @visitor(Norm)
    def _visit_norm(self, node: Norm) -> cp.Expression:
        """Lower norm operation to CVXPy expression.

        Norms are convex functions and commonly used in convex optimization.
        Supports all CVXPy norm types (1, 2, inf, "fro", etc.).

        Args:
            node: Norm expression node with ord attribute

        Returns:
            CVXPy expression representing the norm of the operand

        Note:
            Common norms: ord=2 (Euclidean), ord=1 (Manhattan), ord="inf"
        """
        operand = self.lower(node.operand)
        return cp.norm(operand, node.ord)

    @visitor(Index)
    def _visit_index(self, node: Index) -> cp.Expression:
        """Lower indexing/slicing operation to CVXPy expression.

        Indexing preserves DCP properties (indexing into convex is convex).

        Args:
            node: Index expression node

        Returns:
            CVXPy expression representing base[index]
        """
        base = self.lower(node.base)
        return base[node.index]

    @visitor(Concat)
    def _visit_concat(self, node: Concat) -> cp.Expression:
        """Lower concatenation to CVXPy expression.

        Concatenates expressions horizontally along axis 0. Scalars are
        promoted to 1D arrays before concatenation. Preserves DCP properties.

        Args:
            node: Concat expression node

        Returns:
            CVXPy expression representing horizontal concatenation

        Note:
            Uses cp.hstack for concatenation. Scalars are reshaped to (1,).
        """
        exprs = [self.lower(child) for child in node.exprs]
        # Ensure all expressions are at least 1D for concatenation
        exprs_1d = []
        for expr in exprs:
            if expr.ndim == 0:  # scalar
                exprs_1d.append(cp.reshape(expr, (1,), order="C"))
            else:
                exprs_1d.append(expr)
        return cp.hstack(exprs_1d)

    @visitor(Sin)
    def _visit_sin(self, node: Sin) -> cp.Expression:
        """Raise NotImplementedError for sine function.

        Sine is not DCP-compliant in CVXPy as it is neither convex nor concave.

        Args:
            node: Sin expression node

        Raises:
            NotImplementedError: Always raised since sine is not DCP-compliant

        Note:
            For constraints involving trigonometric functions:
            - Use piecewise-linear approximations, or
            - Handle in the JAX dynamics/constraint layer instead of CVXPy
        """
        raise NotImplementedError(
            "Trigonometric functions like Sin are not DCP-compliant in CVXPy. "
            "Consider using piecewise-linear approximations or handle these constraints "
            "in the dynamics (JAX) layer instead."
        )

    @visitor(Cos)
    def _visit_cos(self, node: Cos) -> cp.Expression:
        """Raise NotImplementedError for cosine function.

        Cosine is not DCP-compliant in CVXPy as it is neither convex nor concave.

        Args:
            node: Cos expression node

        Raises:
            NotImplementedError: Always raised since cosine is not DCP-compliant

        Note:
            For constraints involving trigonometric functions:
            - Use piecewise-linear approximations, or
            - Handle in the JAX dynamics/constraint layer instead of CVXPy
        """
        raise NotImplementedError(
            "Trigonometric functions like Cos are not DCP-compliant in CVXPy. "
            "Consider using piecewise-linear approximations or handle these constraints "
            "in the dynamics (JAX) layer instead."
        )

    @visitor(Tan)
    def _visit_tan(self, node: Tan) -> cp.Expression:
        """Raise NotImplementedError for tangent function.

        Tangent is not DCP-compliant in CVXPy as it is neither convex nor concave.

        Args:
            node: Tan expression node

        Raises:
            NotImplementedError: Always raised since tangent is not DCP-compliant

        Note:
            For constraints involving trigonometric functions:
            - Use piecewise-linear approximations, or
            - Handle in the JAX dynamics/constraint layer instead of CVXPy
        """
        raise NotImplementedError(
            "Trigonometric functions like Tan are not DCP-compliant in CVXPy. "
            "Consider using piecewise-linear approximations or handle these constraints "
            "in the dynamics (JAX) layer instead."
        )

    @visitor(All)
    def _visit_all(self, node: All) -> cp.Expression:
        """Raise NotImplementedError for All expression.

        Logical AND reduction (All) is not DCP-compliant in CVXPy as it introduces
        non-convex boolean logic. All expressions are only supported in JAX lowering.

        Args:
            node: All expression node

        Raises:
            NotImplementedError: Always raised since logical reductions are not DCP-compliant
        """
        raise NotImplementedError(
            "Logical reduction expressions (All) are not DCP-compliant in CVXPy. "
            "Logical operations are only supported in JAX lowering."
        )

    @visitor(Any)
    def _visit_any(self, node: Any) -> cp.Expression:
        """Raise NotImplementedError for Any expression.

        Logical OR reduction (Any) is not DCP-compliant in CVXPy as it introduces
        non-convex boolean logic. Any expressions are only supported in JAX lowering.

        Args:
            node: Any expression node

        Raises:
            NotImplementedError: Always raised since logical reductions are not DCP-compliant
        """
        raise NotImplementedError(
            "Logical reduction expressions (Any) are not DCP-compliant in CVXPy. "
            "Logical operations are only supported in JAX lowering."
        )

    @visitor(Cond)
    def _visit_cond(self, node: Cond) -> cp.Expression:
        """Raise NotImplementedError for conditional expression.

        Conditional logic (Cond) is not DCP-compliant in CVXPy as it introduces
        non-convex branching behavior. Conditional expressions are only supported
        in JAX lowering for dynamics and non-convex constraints.

        Args:
            node: Cond expression node

        Raises:
            NotImplementedError: Always raised since conditional logic is not DCP-compliant

        Note:
            For conditional constraints:
            - Use piecewise-linear approximations, or
            - Handle in the JAX dynamics/constraint layer instead of CVXPy
        """
        raise NotImplementedError(
            "Conditional expressions (Cond) are not DCP-compliant in CVXPy. "
            "Conditional logic is only supported in JAX lowering. Consider using "
            "piecewise-linear approximations or handle these constraints in the "
            "dynamics (JAX) layer instead."
        )

    @visitor(Exp)
    def _visit_exp(self, node: Exp) -> cp.Expression:
        """Lower exponential function to CVXPy expression.

        Exponential is a convex function and DCP-compliant when used in
        appropriate contexts (e.g., minimizing exp(x) or constraints like
        exp(x) <= c).

        Args:
            node: Exp expression node

        Returns:
            CVXPy expression representing exp(operand)

        Note:
            Exponential is convex increasing, so it's valid in:
            - Objective: minimize exp(x)
            - Constraints: exp(x) <= c (convex constraint)
        """
        operand = self.lower(node.operand)
        return cp.exp(operand)

    @visitor(Log)
    def _visit_log(self, node: Log) -> cp.Expression:
        """Lower natural logarithm to CVXPy expression.

        Logarithm is a concave function and DCP-compliant when used in
        appropriate contexts (e.g., maximizing log(x) or constraints like
        log(x) >= c).

        Args:
            node: Log expression node

        Returns:
            CVXPy expression representing log(operand)

        Note:
            Logarithm is concave increasing, so it's valid in:
            - Objective: maximize log(x)
            - Constraints: log(x) >= c (concave constraint, or equivalently c <= log(x))
        """
        operand = self.lower(node.operand)
        return cp.log(operand)

    @visitor(Abs)
    def _visit_abs(self, node: Abs) -> cp.Expression:
        """Lower absolute value to CVXPy expression.

        Absolute value is a convex function and DCP-compliant when used in
        appropriate contexts (e.g., minimizing |x| or constraints like |x| <= c).

        Args:
            node: Abs expression node

        Returns:
            CVXPy expression representing |operand|

        Note:
            Absolute value is convex, so it's valid in:
            - Objective: minimize abs(x)
            - Constraints: abs(x) <= c (convex constraint)
        """
        operand = self.lower(node.operand)
        return cp.abs(operand)

    @visitor(Equality)
    def _visit_equality(self, node: Equality) -> cp.Constraint:
        """Lower equality constraint to CVXPy constraint (lhs == rhs).

        Equality constraints require affine expressions on both sides for
        DCP compliance.

        Args:
            node: Equality constraint node

        Returns:
            CVXPy equality constraint object

        Note:
            For DCP compliance, both lhs and rhs must be affine. CVXPy will
            raise a DCP error if either side is non-affine.
        """
        left = self.lower(node.lhs)
        right = self.lower(node.rhs)
        return left == right

    @visitor(Inequality)
    def _visit_inequality(self, node: Inequality) -> cp.Constraint:
        """Lower inequality constraint to CVXPy constraint (lhs <= rhs).

        Inequality constraints must satisfy DCP rules: convex <= concave.

        Args:
            node: Inequality constraint node

        Returns:
            CVXPy inequality constraint object

        Note:
            For DCP compliance: lhs must be convex and rhs must be concave.
            Common form: convex_expr(x) <= constant
        """
        left = self.lower(node.lhs)
        right = self.lower(node.rhs)
        return left <= right

    @visitor(CTCS)
    def _visit_ctcs(self, node: CTCS) -> cp.Expression:
        """Raise NotImplementedError for CTCS constraints.

        CTCS (Continuous-Time Constraint Satisfaction) constraints are handled
        through dynamics augmentation using JAX, not CVXPy. They represent
        non-convex continuous-time constraints.

        Args:
            node: CTCS constraint node

        Raises:
            NotImplementedError: Always raised since CTCS uses JAX, not CVXPy

        Note:
            CTCS constraints are lowered to JAX during dynamics augmentation.
            They add virtual states and controls to enforce constraints over
            continuous time intervals. See JaxLowerer.visit_ctcs() instead.
        """
        raise NotImplementedError(
            "CTCS constraints are for continuous-time constraint satisfaction and "
            "should be handled through dynamics augmentation with JAX lowering, "
            "not CVXPy lowering. CTCS constraints represent non-convex dynamics "
            "augmentation."
        )

    @visitor(PositivePart)
    def _visit_pos(self, node: PositivePart) -> cp.Expression:
        """Lower positive part function to CVXPy.

        Computes max(x, 0), which is convex. Used in penalty methods for
        inequality constraints.

        Args:
            node: PositivePart expression node

        Returns:
            CVXPy expression representing max(operand, 0)

        Note:
            Positive part is convex and commonly used in hinge loss and
            penalty methods for inequality constraints.
        """
        operand = self.lower(node.x)
        return cp.maximum(operand, 0.0)

    @visitor(Square)
    def _visit_square(self, node: Square) -> cp.Expression:
        """Lower square function to CVXPy.

        Computes x^2, which is convex. Used in quadratic penalty methods
        and least-squares objectives.

        Args:
            node: Square expression node

        Returns:
            CVXPy expression representing operand^2

        Note:
            Square is convex increasing for x >= 0 and convex decreasing for
            x <= 0. It's always convex overall.
        """
        operand = self.lower(node.x)
        return cp.square(operand)

    @visitor(Huber)
    def _visit_huber(self, node: Huber) -> cp.Expression:
        """Lower Huber penalty function to CVXPy.

        Huber penalty is quadratic for small values and linear for large values,
        providing robustness to outliers. It is convex and DCP-compliant.

        The Huber function is defined as:
        - |x| <= delta: 0.5 * x^2
        - |x| > delta: delta * (|x| - 0.5 * delta)

        Args:
            node: Huber expression node with delta parameter

        Returns:
            CVXPy expression representing Huber penalty

        Note:
            Huber loss is convex and combines the benefits of squared error
            (smooth, differentiable) and absolute error (robust to outliers).
        """
        operand = self.lower(node.x)
        return cp.huber(operand, M=node.delta)

    @visitor(SmoothReLU)
    def _visit_srelu(self, node: SmoothReLU) -> cp.Expression:
        """Lower smooth ReLU penalty function to CVXPy.

        Smooth approximation to ReLU: sqrt(max(x, 0)^2 + c^2) - c
        Differentiable everywhere, approaches ReLU as c -> 0. Convex.

        Args:
            node: SmoothReLU expression node with smoothing parameter c

        Returns:
            CVXPy expression representing smooth ReLU penalty

        Note:
            This provides a smooth, convex approximation to the ReLU function
            max(x, 0). The parameter c controls the smoothness: smaller c gives
            a better approximation but less smoothness.
        """
        operand = self.lower(node.x)
        c = node.c
        # smooth_relu(x) = sqrt(max(x, 0)^2 + c^2) - c
        pos_part = cp.maximum(operand, 0.0)
        # For SmoothReLU, we use the 2-norm formulation
        return cp.sqrt(cp.sum_squares(pos_part) + c**2) - c

    @visitor(Sqrt)
    def _visit_sqrt(self, node: Sqrt) -> cp.Expression:
        """Lower square root to CVXPy expression.

        Square root is concave and DCP-compliant when used appropriately
        (e.g., maximizing sqrt(x) or constraints like sqrt(x) >= c).

        Args:
            node: Sqrt expression node

        Returns:
            CVXPy expression representing sqrt(operand)

        Note:
            Square root is concave increasing for x > 0. Valid in:
            - Objective: maximize sqrt(x)
            - Constraints: sqrt(x) >= c (concave constraint)
        """
        operand = self.lower(node.operand)
        return cp.sqrt(operand)

    @visitor(Max)
    def _visit_max(self, node: Max) -> cp.Expression:
        """Lower element-wise maximum to CVXPy expression.

        Maximum is convex (pointwise max of convex functions is convex).

        Args:
            node: Max expression node with multiple operands

        Returns:
            CVXPy expression representing element-wise maximum

        Note:
            For multiple operands, chains binary maximum operations.
            Maximum preserves convexity.
        """
        operands = [self.lower(op) for op in node.operands]
        # CVXPy's maximum can take multiple arguments
        if len(operands) == 2:
            return cp.maximum(operands[0], operands[1])
        else:
            # For more than 2 operands, chain maximum calls
            result = cp.maximum(operands[0], operands[1])
            for op in operands[2:]:
                result = cp.maximum(result, op)
            return result

    @visitor(LogSumExp)
    def _visit_logsumexp(self, node: LogSumExp) -> cp.Expression:
        """Lower log-sum-exp to CVXPy expression.

        Log-sum-exp is convex and is a smooth approximation to the maximum function.
        CVXPy's log_sum_exp atom computes log(sum(exp(x_i))) for stacked operands.

        Args:
            node: LogSumExp expression node with multiple operands

        Returns:
            CVXPy expression representing log-sum-exp

        Note:
            Log-sum-exp is convex and DCP-compliant. It satisfies:
            max(x₁, ..., xₙ) ≤ logsumexp(x₁, ..., xₙ) ≤ max(x₁, ..., xₙ) + log(n)
        """
        operands = [self.lower(op) for op in node.operands]

        # CVXPy's log_sum_exp expects a stacked expression with an axis parameter
        # For element-wise log-sum-exp, we stack along a new axis and reduce along it
        if len(operands) == 1:
            return operands[0]

        # Stack operands along a new axis (axis 0) and compute log_sum_exp along that axis
        stacked = cp.vstack(operands)
        return cp.log_sum_exp(stacked, axis=0)

    @visitor(Transpose)
    def _visit_transpose(self, node: Transpose) -> cp.Expression:
        """Lower matrix transpose to CVXPy expression.

        Transpose preserves DCP properties (transpose of convex is convex).

        Args:
            node: Transpose expression node

        Returns:
            CVXPy expression representing operand.T
        """
        operand = self.lower(node.operand)
        return operand.T

    @visitor(Inv)
    def _visit_inv(self, node: Inv) -> cp.Expression:
        """Raise NotImplementedError for matrix inverse.

        Matrix inverse is not DCP-compliant in CVXPy as it is neither convex
        nor concave for variable matrices.

        Args:
            node: Inv expression node

        Raises:
            NotImplementedError: Always raised since matrix inverse is not DCP-compliant

        Note:
            For optimization problems requiring matrix inverse:
            - If the matrix is constant/parameter, compute the inverse numerically
              before passing to CVXPy
            - Handle matrix inverse in the JAX dynamics/constraint layer instead
            - Consider reformulating the problem to avoid explicit matrix inverse
        """
        raise NotImplementedError(
            "Matrix inverse (Inv) is not DCP-compliant in CVXPy. "
            "inv(X) is neither convex nor concave for variable matrices. "
            "Consider: (1) computing the inverse numerically if the matrix is constant, "
            "(2) handling this in the JAX layer instead, or "
            "(3) reformulating the problem to avoid explicit matrix inverse."
        )

    @visitor(Power)
    def _visit_power(self, node: Power) -> cp.Expression:
        """Lower element-wise power (base**exponent) to CVXPy expression.

        Power is DCP-compliant for specific exponent values:
        - exponent >= 1: convex (when base >= 0)
        - 0 <= exponent <= 1: concave (when base >= 0)

        Args:
            node: Power expression node

        Returns:
            CVXPy expression representing base**exponent

        Note:
            CVXPy will verify DCP compliance at problem construction time.
            Common convex cases: x^2, x^3, x^4 (even powers)
        """
        base = self.lower(node.base)
        exponent = self.lower(node.exponent)
        return cp.power(base, exponent)

    @visitor(Stack)
    def _visit_stack(self, node: Stack) -> cp.Expression:
        """Lower vertical stacking to CVXPy expression.

        Stacks expressions vertically using cp.vstack. Preserves DCP properties.

        Args:
            node: Stack expression node with multiple rows

        Returns:
            CVXPy expression representing vertical stack of rows

        Note:
            Each row is stacked along axis 0 to create a 2D array.
        """
        rows = [self.lower(row) for row in node.rows]
        # Stack rows vertically
        return cp.vstack(rows)

    @visitor(Hstack)
    def _visit_hstack(self, node: Hstack) -> cp.Expression:
        """Lower horizontal stacking to CVXPy expression.

        For 1D arrays, uses cp.hstack (concatenation). For 2D+ arrays, uses
        cp.bmat with a single row to achieve proper horizontal stacking along
        axis 1, matching numpy.hstack semantics.

        Args:
            node: Hstack expression node with multiple arrays

        Returns:
            CVXPy expression representing horizontal stack of arrays
        """
        arrays = [self.lower(arr) for arr in node.arrays]

        # Check dimensionality from the symbolic node's shape
        shape = node.check_shape()
        if len(shape) == 1:
            # 1D: simple concatenation
            return cp.hstack(arrays)
        else:
            # 2D+: use bmat with single row for proper horizontal stacking
            return cp.bmat([arrays])

    @visitor(Vstack)
    def _visit_vstack(self, node: Vstack) -> cp.Expression:
        """Lower vertical stacking to CVXPy expression.

        Stacks expressions vertically using cp.vstack. Preserves DCP properties.

        Args:
            node: Vstack expression node with multiple arrays

        Returns:
            CVXPy expression representing vertical stack of arrays
        """
        arrays = [self.lower(arr) for arr in node.arrays]
        return cp.vstack(arrays)

    @visitor(Block)
    def _visit_block(self, node: Block) -> cp.Expression:
        """Lower block matrix construction to CVXPy expression.

        Assembles a block matrix from nested lists of expressions using cp.bmat.
        This is the CVXPy equivalent of numpy.block() for block matrix construction.

        Args:
            node: Block expression node with 2D nested structure of expressions

        Returns:
            CVXPy expression representing the assembled block matrix

        Raises:
            NotImplementedError: If any block has more than 2 dimensions

        Note:
            cp.bmat preserves DCP properties when all blocks are DCP-compliant.
            Block matrices are commonly used for constraint aggregation.
            For 3D+ tensors, use JAX lowering instead.
        """
        # Check for 3D+ blocks - CVXPy's bmat only supports 2D
        for i, row in enumerate(node.blocks):
            for j, block in enumerate(row):
                block_shape = block.check_shape()
                if len(block_shape) > 2:
                    raise NotImplementedError(
                        f"CVXPy does not support Block with tensors of dimension > 2. "
                        f"Block[{i}][{j}] has shape {block_shape} ({len(block_shape)}D). "
                        f"For N-D tensor block assembly, use JAX lowering instead."
                    )

        # Lower each block expression
        block_exprs = [[self.lower(block) for block in row] for row in node.blocks]
        return cp.bmat(block_exprs)

    @visitor(Linterp)
    def _visit_linterp(self, node: Linterp) -> cp.Expression:
        """Raise NotImplementedError for linear interpolation.

        Linear interpolation (Linterp) is not DCP-compliant in CVXPy as it
        represents a piecewise-linear function that is neither convex nor
        concave in general.

        Args:
            node: Linterp expression node

        Raises:
            NotImplementedError: Always raised since Linterp is not DCP-compliant
        """
        raise NotImplementedError("Linear interpolation (Linterp) is not DCP-compliant in CVXPy.")

    @visitor(Bilerp)
    def _visit_bilerp(self, node: Bilerp) -> cp.Expression:
        """Raise NotImplementedError for bilinear interpolation.

        Bilinear interpolation (Bilerp) is not DCP-compliant in CVXPy as it
        represents a nonlinear function that is neither convex nor concave.

        Args:
            node: Bilerp expression node

        Raises:
            NotImplementedError: Always raised since Bilerp is not DCP-compliant
        """
        raise NotImplementedError("Bilinear interpolation (Bilerp) is not DCP-compliant in CVXPy.")


def lower_to_cvxpy(expr: Expr, variable_map: Dict[str, cp.Expression] = None) -> cp.Expression:
    """Lower symbolic expression to CVXPy expression or constraint.

    Convenience wrapper that creates a CvxpyLowerer and lowers a single
    symbolic expression to a CVXPy expression. The result can be used in
    CVXPy optimization problems.

    Args:
        expr: Symbolic expression to lower (any Expr subclass)
        variable_map: Dictionary mapping variable names to CVXPy expressions.
            Must include "x" for states and "u" for controls. May include
            parameter names mapped to CVXPy Parameters or constants.

    Returns:
        CVXPy expression for arithmetic expressions (Add, Mul, Norm, etc.)
        or CVXPy constraint for constraint expressions (Equality, Inequality)

    Raises:
        NotImplementedError: If the expression type is not supported (e.g., Sin, Cos, CTCS)
        ValueError: If required variables are missing from variable_map

    Example:
        Basic expression lowering::

            import cvxpy as cp
            import openscvx as ox

            # Create CVXPy variables
            cvx_x = cp.Variable(3, name="x")
            cvx_u = cp.Variable(2, name="u")

            # Create symbolic expression
            x = ox.State("x", shape=(3,))
            u = ox.Control("u", shape=(2,))
            expr = ox.Norm(x)**2 + 0.1 * ox.Norm(u)**2

            # Lower to CVXPy
            cvx_expr = lower_to_cvxpy(expr, {"x": cvx_x, "u": cvx_u})

            # Use in optimization problem
            prob = cp.Problem(cp.Minimize(cvx_expr))
            prob.solve()

        Constraint lowering::

            # Symbolic constraint
            constraint = ox.Norm(x) <= 1.0

            # Lower to CVXPy constraint
            cvx_constraint = lower_to_cvxpy(constraint, {"x": cvx_x, "u": cvx_u})

            # Use in problem
            prob = cp.Problem(cp.Minimize(cost), constraints=[cvx_constraint])

    See Also:
        - CvxpyLowerer: The underlying lowerer class
        - lower_to_jax(): Convenience wrapper for JAX lowering
        - lower_symbolic_expressions(): Main orchestrator in symbolic/lower.py
    """
    lowerer = CvxpyLowerer(variable_map)
    return lowerer.lower(expr)
