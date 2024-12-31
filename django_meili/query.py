# pylint: disable=W0719,E1101,C0103,E1131
"""
query.py < Leo >
A class for constructing complex query expressions for MeiliSearch.

=================
Query (Q) Module
This module provides the Q class, which is designed to help create
complex query expressions for searching documents in MeiliSearch
indices. The Q class allows the building queries using logical operators
(AND, OR, NOT) and various comparison operators
(EQUALS, NOT_EQUALS, GREATER_THAN, etc.).
You can use this class to create MeiliSearch queries that filter and
sort search results based on specific conditions.

Classes:
- Q: A class for constructing complex query expressions for MeiliSearch.

Usage example:
```from query import Q

# Create a Q object with a single condition
q = Q(title__eq='The Catcher in the Rye')

# Combine conditions using logical operators
q = Q(title__eq='The Catcher in the Rye') | Q(author__eq='J.D. Salinger')

# Negate conditions using the ~ operator
q = ~Q(title__eq='The Catcher in the Rye')

# Generate a MeiliSearch query string from the Q object
query_string = q.to_query_string()

# "Complex" queries with characters not allowed by python kwargs
q = Q(**{'category.name__eq': 'Fiction', 'year__gte': 1950})
```
"""
from enum import Enum
from typing import Any, List, Union


class Q:
    """Encapsulates a set of comparison operations and logical constructs for building
    query conditions, providing features like combining operations using logical
    operators (AND, OR), negation, and validation of query structures. Primarily
    designed for constructing queries compatible with MeiliSearch.

    Attributes:
        conditions (dict): A dictionary holding the field-condition pairs that define
            the logic of the query.
        operator (str, optional): Logical operator (e.g., 'OR', 'AND') used to combine
            query clauses. None if the object represents only one clause.
        negate (bool): A flag indicating whether the query logic is negated.
        children (list): Nested child `Q` objects representing subqueries combined
            using logical operators.
    """

    class OPERATIONS(str, Enum):
        """
        Represents operations as an enumeration of string constants.

        This class is a subclass of Enum, providing a set of predefined string constants
        that represent comparison and existence-check operations. These operations can
        be used in filtering, querying, or other contexts requiring such operations.

        Attributes:
            EQUALS (str): Represents the "equal to" operation.
            NOT_EQUALS (str): Represents the "not equal to" operation.
            GREATER_THAN (str): Represents the "greater than" operation.
            GREATER_THAN_OR_EQUALS (str): Represents the "greater than or equal to" operation.
            LESS_THAN (str): Represents the "less than" operation.
            LESS_THAN_OR_EQUALS (str): Represents the "less than or equal to" operation.
            IN (str): Represents the "in" operation for set membership.
            NOT_IN (str): Represents the "not in" operation for set membership.
            EXISTS (str): Represents the "exists" operation for checking presence.
            NOT_EXISTS (str): Represents the "not exists" operation for checking absence.
        """

        EQUALS = "eq"
        NOT_EQUALS = "neq"
        GREATER_THAN = "gt"
        GREATER_THAN_OR_EQUALS = "gte"
        LESS_THAN = "lt"
        LESS_THAN_OR_EQUALS = "lte"
        IN = "in"
        NOT_IN = "nin"
        EXISTS = "exists"
        NOT_EXISTS = "not exists"

        @classmethod
        def get_values(cls):
            """
            Retrieves a list of values from the `_member_map_` attribute of the class.

            This method accesses the `_member_map_` attribute of the class, which is expected to
            hold a mapping of members, and extracts a list of the corresponding values.

            Returns:
                list: A list of values from the `_member_map_` attribute.
            """
            return list(cls._member_map_.values())

    op_map = {
        OPERATIONS.EQUALS: "=",
        OPERATIONS.NOT_EQUALS: "!=",
        OPERATIONS.GREATER_THAN: ">",
        OPERATIONS.GREATER_THAN_OR_EQUALS: ">=",
        OPERATIONS.LESS_THAN: "<",
        OPERATIONS.LESS_THAN_OR_EQUALS: "<=",
        OPERATIONS.IN: "IN",
        OPERATIONS.NOT_IN: "NOT IN",
        OPERATIONS.EXISTS: "EXISTS",
        OPERATIONS.NOT_EXISTS: "NOT EXISTS",
    }
    negate_map = {
        "=": "!=",
        "!=": "=",
        ">": "<=",
        ">=": "<",
        "<": ">=",
        "<=": ">",
        op_map[OPERATIONS.IN]: op_map[OPERATIONS.NOT_IN],
        op_map[OPERATIONS.NOT_IN]: op_map[OPERATIONS.IN],
        op_map[OPERATIONS.EXISTS]: op_map[OPERATIONS.NOT_EXISTS],
        op_map[OPERATIONS.NOT_EXISTS]: op_map[OPERATIONS.EXISTS],
    }

    def __init__(self, **kwargs) -> None:
        """
        Represents a logic combination using conditions, logical operators, and children
        sub-expressions. Allows the configuration of a variety of logical operations and
        conditions to build complex logical expressions.

        Args:
            **kwargs: Arbitrary keyword arguments representing conditions to initialize
                the object with.

        Attributes:
            conditions: A dictionary holding the conditions provided during instantiation.
            operator: A logical operator (e.g., AND, OR) used to combine conditions or
                child expressions. Defaults to None.
            negate: A boolean indicating whether the entire logical expression should
                be negated. Defaults to False.
            children: A list of child logical expressions or conditions that compose
                the overall logical structure. Defaults to an empty list.
        """
        self.conditions = kwargs
        self.operator = None
        self.negate = False
        self.children = []

    def __or__(self, other) -> "Q":
        """
        Performs a logical OR operation between two query objects and combines their children.

        This method creates a new query object that represents the OR operation
        between the current query object and another query object. If either
        query is empty, the non-empty query will be returned. If both are non-empty,
        a combined query object is created with an OR operator and a list of both
        queries as children.

        Args:
            other: A query object of type Q to be combined with the current query
                object using an OR operation.

        Returns:
            Q: A new query object representing the logical OR operation between
            the two query objects, or one of the query objects if the other is
            empty.
        """
        if self.is_empty():
            return other
        if other.is_empty():
            return self

        new_q = Q()
        new_q.operator = "OR"
        new_q.children = [self, other]
        return new_q

    def __and__(self, other) -> "Q":
        """
        Represents the logical AND operation between two Q objects. Combines the conditions
        defined in two Q objects into a single Q object using the "AND" operator. This allows
        building more complex query structures in a compositional manner.

        Args:
            other (Q): A Q object to be combined with the current Q object using the "AND" operator.

        Returns:
            Q: A new Q object representing the combined conditions of the current Q object and
               the input Q object using the "AND" operator.
        """
        if self.is_empty():
            return other
        if other.is_empty():
            return self

        new_q = Q()
        new_q.operator = "AND"
        new_q.children = [self, other]
        return new_q

    def __invert__(self) -> "Q":
        """
        Inverts the logic of the query object by toggling its negate attribute.

        This method creates a new instance of `Q` that mirrors the current query
        object but with the logical negation applied. The other attributes such
        as conditions, operator, and children are directly copied to the new
        instance.

        Returns:
            Q: A new instance of the `Q` class with the logic negated.
        """
        new_q = Q()
        new_q.negate = not self.negate
        new_q.conditions = self.conditions
        new_q.operator = self.operator
        new_q.children = self.children
        return new_q

    def _clean_value(self, val) -> Union[str, int, dict, List[Any]]:
        """
        Cleans and formats a given value based on its type. The function prepares the value
        for safe use in operations or further processing by implementing specific formatting
        rules for strings, booleans, and other types. Strings containing spaces or matching
        specific operation keywords are wrapped in double quotes, booleans are converted to
        lowercase strings, and other types are converted to strings.

        Args:
            val: Input value to be processed. It can be of type `str`, `bool`, or other types
                which are convertible to strings.

        Returns:
            Union[str, int, dict, List[Any]]: The processed value in its formatted form,
            suitable for operations. Depending on the type of input value, the return value
            may vary, with strings and booleans receiving special processing.
        """
        retval = val
        if isinstance(val, str):
            if " " in val:
                retval = f'"{val}"'
            else:
                retval = val
        elif isinstance(val, bool):
            retval = str(val).lower()
        else:
            retval = str(val)
        expression_ops = list(self.OPERATIONS.get_values()) + list(
            self.negate_map.values()
        )
        if retval in expression_ops:
            retval = f'"{retval}"'
        return retval

    def __repr__(self) -> str:
        """
        Represents the object's string representation used for debugging and logging.

        This method returns a string that provides a textual representation of
        the object, which is useful for developers to understand the current
        state of the object. The string is constructed using the `to_query_string`
        method, encapsulated in a specific format.

        Returns:
            str: A string representation of the object.
        """
        return f"Q({self.to_query_string()})"

    def is_empty(self) -> bool:
        """
        Determines if the current object is considered empty.

        An object is regarded as empty if it has no conditions and either has no children
        or all its children are also empty. This method recursively checks each child
        to determine emptiness.

        Returns:
            bool: True if the object is empty; False otherwise.
        """
        if self.conditions:
            return False
        if not self.children:
            return True
        return all(child.is_empty() for child in self.children)

    def to_query_string(self) -> str:
        """
        Converts the current object instance into a query string representation.

        The method recursively processes the children nodes if an operator is present
        and assembles conditions into query string format. It supports logical operators
        (AND/OR) and a variety of field operations (EQUALS, IN, NOT_IN, EXISTS, NOT EXISTS).
        Conditions and field operations are sanitized and validated to ensure correctness
        of query building logic.

        Returns:
            str: A query string representation of the current instance.

        Raises:
            ValueError: If an invalid operation is provided in the conditions.
            ValueError: If the value of an IN or NOT_IN operation is not a list.
        """
        if self.is_empty():
            return ""

        if self.operator:
            # Get query strings for both children
            left = self.children[0].to_query_string()
            right = self.children[1].to_query_string()

            # Handle cases where one or both children might be empty
            if not left:
                return right
            if not right:
                return left

            # Only add parentheses if there are multiple conditions
            left_needs_parens = self.children[0].operator is not None
            right_needs_parens = self.children[1].operator is not None

            left_str = f"({left})" if left_needs_parens else left
            right_str = f"({right})" if right_needs_parens else right

            return f"{left_str} {self.operator} {right_str}"

        conditions = []
        for key, value in self.conditions.items():
            *fields, operation = (
                key.split("__") if "__" in key else (key, self.OPERATIONS.EQUALS)
            )
            field = ".".join(fields)
            assert operation in self.OPERATIONS.get_values(), ValueError(
                f"Invalid operation {operation}"
            )

            if operation == self.OPERATIONS.IN:
                assert isinstance(value, list), ValueError(
                    f"Value for IN operation must be a list. Got {value}"
                )
                escaped_values = [self._clean_value(i) for i in value]
                value = f"[{','.join(escaped_values)}]"
            elif operation == self.OPERATIONS.NOT_IN:
                assert isinstance(value, list), ValueError(
                    f"Value for NOT_IN operation must be a list. Got {value}"
                )
                escaped_values = [self._clean_value(i) for i in value]
                value = f"[{','.join(escaped_values)}]"
            else:
                value = self._clean_value(value)

            if self.negate:
                operation = self.negate_map[self.op_map[operation]]
            else:
                operation = self.op_map[operation]

            if operation in ["EXISTS", "NOT EXISTS"]:
                condition = f"{field} {operation}"
            else:
                condition = f"{field} {operation} {value}"
            conditions.append(condition)

        return " AND ".join(conditions) if conditions else ""

    def to_query_list(self, lvl: int = 0) -> list:
        """
        Converts a query object into a list of query strings or subqueries recursively for use
        with Meilisearch. Handles nested query structures, conditions, and supported operations.

        Args:
            lvl (int): The level of nesting for the query. Defaults to 0. Determines the
                depth of the query building process. Meilisearch only supports up to
                2 levels of nesting, so an exception will be raised if the nesting
                exceeds this.

        Raises:
            Exception: Raised if the nesting level exceeds 2, as Meilisearch does not support
                deeper nested queries.
            AssertionError: Raised if:
                - An invalid operation is provided that is not supported by the
                  defined set of operations.
                - The provided value for 'IN' or 'NOT_IN' operations is not a list.

        Returns:
            list: A list representing the constructed query string(s) or subqueries,
                depending on the structure of the query object.
        """
        if lvl > 2:
            raise ValueError(
                "Query nesting too deep, meilisearch only supports 2 levels of nesting"
            )
        if self.operator:
            left = self.children[0].to_query_list(lvl + 1)
            right = self.children[1].to_query_list(lvl + 1)
            return [left, right]
        conditions = []
        for key, value in self.conditions.items():
            *fields, operation = (
                key.split("__") if "__" in key else (key, self.OPERATIONS.EQUALS)
            )
            field = ".".join(fields)
            assert (
                    operation in self.OPERATIONS.get_values()
            ), f"Invalid operation {operation}"
            if operation == self.OPERATIONS.IN:
                assert isinstance(
                    value, list
                ), f"Value for IN operation must be a list. Got {value}"
                escaped_values = [self._clean_value(i) for i in value]
                value = f"[{','.join(escaped_values)}]"
            elif operation == self.OPERATIONS.NOT_IN:
                assert isinstance(
                    value, list
                ), f"Value for NOT_IN operation must be a list. Got {value}"
                escaped_values = [self._clean_value(i) for i in value]
                value = f"[{','.join(escaped_values)}]"
            else:
                if isinstance(value, str) and " " in value:
                    value = f'"{value}"'
            if self.negate:
                operation = self.negate_map[self.op_map[operation]]
            else:
                operation = self.op_map[operation]
            if operation in ["EXISTS", "NOT EXISTS"]:
                condition = f"{field} {operation}"
            else:
                condition = f"{field} {operation} {value}"
            conditions.append(condition)
        return " AND ".join(conditions)  # type: ignore

    def prettify_query_string(self) -> str:
        """
        Formats a query string into a more human-readable form with proper indentation.

        The function processes a query string created from the `to_query_string` method
        and formats it with indentation for improved readability. Logical operators
        such as "AND" and "OR" are placed on new lines with the appropriate indentation
        level. The function utilizes a stack to track characters and adjust the
        indentation level dynamically based on opening and closing parentheses.

        Returns:
            str: A prettified and more readable version of the query string.

        Raises:
            None
        """
        query_string = self.to_query_string()
        stack = []
        result = []
        indent = 0
        space = "    "

        for char in query_string:
            if char == "(":
                stack.append(char)
                indent += 1
                result.append(char)
            elif char == ")":
                stack.pop()
                indent -= 1
                result.append(char)
            elif char in ["A", "O"] and "".join(stack[-3:]) in ["AND", "OR "]:
                result.append("\n" + indent * space + char)
                stack.pop()
            else:
                stack.append(char)
                result.append(char)

            if len(stack) >= 3 and "".join(stack[-3:]) in ["AND", "OR "]:
                result.append("\n" + indent * space)

        return "".join(result)

    def explain(self, indent_level: int = 0) -> str:
        """
        Generates an explanation string for a logical or conditional construct, detailing its
        structure and conditions. The explanation is nicely formatted with the specified
        indentation levels for enhanced readability.

        Args:
            indent_level (int): The current level of indentation used to format the explanation
                output. Defaults to 0.

        Returns:
            str: A formatted explanation string representing either a logical operation with
                operands explained recursively or conditions with their respective explanations.

        Raises:
            KeyError: Raised when an operation specified in conditions is not present in
                the mapping of operations.
        """

        def indent(text: str, level: int) -> str:
            return "    " * level + text

        if self.operator:
            left = self.children[0].explain(indent_level + 1)
            right = self.children[1].explain(indent_level + 1)
            return (
                    f"{indent('BEGIN', indent_level)}\n{left}"
                    + f"\n{indent(self.operator, indent_level)}\n{right}"
            )
        conditions = []
        for key, value in self.conditions.items():
            *fields, operation = (
                key.split("__") if "__" in key else (key, Q.OPERATIONS.EQUALS)
            )
            field = ".".join(fields)
            if self.negate:
                operation = self.negate_map[self.op_map[operation]]
            else:
                operation = self.op_map[operation]
            explanation = ""
            if operation == "EXISTS":
                explanation = "exists"
            elif operation == "NOT EXISTS":
                explanation = "does not exist"
            elif operation == "IN":
                explanation = f"is in {value}"
            elif operation == "NOT IN":
                explanation = f"is not in {value}"
            else:
                explanation = f"is {operation.lower()} {value}"
            conditions.append(
                indent(
                    f'field "{field}":\n{indent(f"* {explanation}", indent_level + 1)}',
                    indent_level,
                )
            )
        return "\n".join(conditions)
