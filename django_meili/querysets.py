"""
querysets.py
Ian Kollipara <ian.kollipara@gmail.com>

This module contains the QuerySet classes for the Django MeiliSearch app.
"""
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
# Imports
from typing import TYPE_CHECKING, Literal, NamedTuple, Self, Type, Dict, Any, Optional, TypeVar, Generic, List
from django.db.models import QuerySet

from ._client import client

if TYPE_CHECKING:
    from .models import IndexMixin


T = TypeVar('T', bound='IndexMixin')

class FilterOperator(Enum):
    """Enum for filter operators to avoid string literals and typos"""
    EXACT = 'exact'
    GTE = 'gte'
    GT = 'gt'
    LTE = 'lte'
    LT = 'lt'
    IN = 'in'
    RANGE = 'range'
    EXISTS = 'exists'
    ISNULL = 'isnull'
    EMPTY = 'empty'

@dataclass(frozen=True)
class SearchOptions:
    """Configuration options for customizing search behavior.

    This class encapsulates various options to tailor the behavior of a search query,
    allowing users to specify what data to retrieve, how to format results, and which
    criteria to use for ranking or filtering results. These options are designed
    to provide fine control over the query execution and result generation processes.

    Attributes:
        vector (Optional[str]): The vector search parameter, used to enable searching
            based on vector similarity.
        hybrid (Optional[str]): The hybrid search parameter, typically combining traditional
            keyword search with vector search.
        offset (Optional[int]): The positional offset for paginating results.
        limit (Optional[int]): The maximum number of results to retrieve.
        page (Optional[int]): The page number for paginated results.
        hits_per_page (Optional[int]): The number of results (hits) to return per page.
        attributes_to_retrieve (Optional[List[str]]): A list of attributes to include in
            the search result for each document.
        retrieve_vectors (Optional[bool]): Whether to include vectors in the search result.
        attributes_to_crop (Optional[List[str]]): A list of attributes from which textual
            content should be cropped for the results.
        crop_length (Optional[int]): The number of characters to include in cropped content.
        attributes_to_highlight (Optional[List[str]]): A list of attributes in which
            matches should be highlighted.
        show_matches_position (Optional[bool]): Whether to include match positions for
            highlighted results.
        show_ranking_score (Optional[bool]): Whether to include the ranking score for
            each result.
        show_ranking_score_details (Optional[bool]): Whether to include detailed
            ranking score information for each result.
        filter (Optional[str]): A filter expression to restrict the search results.
        sort (Optional[List[str]]): A list of sorting criteria to apply to the results.
        distinct (Optional[bool]): Whether to enable the distinct feature, which deduplicates
            similar results.
        facets (Optional[List[str]]): A list of attributes to use as facets in the results.
        highlight_pre_tag (Optional[str]): The string or tag to prepend to highlighted
            matches in the results.
        highlight_post_tag (Optional[str]): The string or tag to append to highlighted
            matches in the results.
        crop_marker (Optional[str]): The marker or string used to indicate cropped sections
            within the results.
        matching_strategy (Optional[Literal["last", "all"]]): Defines the matching strategy
            for search, either "last" or "all".
        attributes_to_search_on (Optional[List[str]]): A list of attributes to specifically
            search within.
        ranking_score_threshold (Optional[int]): The minimum threshold for a ranking score
            that results must meet to be included.
        locales (Optional[List[str]]): A list of locale codes to use for localization during
            the search process.
    """
    vector: Optional[str] = None
    hybrid: Optional[str] = None
    offset: Optional[int] = None
    limit: Optional[int] = None
    page: Optional[int] = None
    hits_per_page: Optional[int] = None
    attributes_to_retrieve: Optional[List[str]] = None
    retrieve_vectors: Optional[bool] = None
    attributes_to_crop: Optional[List[str]] = None
    crop_length: Optional[int] = None
    attributes_to_highlight: Optional[List[str]] = None
    show_matches_position: Optional[bool] = None
    show_ranking_score: Optional[bool] = None
    show_ranking_score_details: Optional[bool] = None
    filter: Optional[str] = None
    sort: Optional[List[str]] = None
    distinct: Optional[bool] = None
    facets: Optional[List[str]] = None
    highlight_pre_tag: Optional[str] = None
    highlight_post_tag: Optional[str] = None
    crop_marker: Optional[str] = None
    matching_strategy: Optional[Literal["last", "all"]] = None
    attributes_to_search_on: Optional[List[str]] = None
    ranking_score_threshold: Optional[int] = None
    locales: Optional[List[str]] = None

@dataclass(frozen=True)
class QuerySetState:
    """
    Represents an immutable state of a query set with various customizable options.

    This class encapsulates query set configurations such as pagination, filters, sorting,
    matching strategy, searchable attributes, and other search-specific options. Designed
    to provide a standardized way to manage and encapsulate query parameters in a frozen data
    structure.

    Attributes:
        offset (int): The starting point for pagination, indicating the number of items to skip.
        limit (int): The maximum number of items to retrieve for pagination purposes.
        filters (tuple[str, ...]): A collection of filter criteria applied to the query,
            typically represented as strings.
        sort (tuple[str, ...]): A collection of fields or properties to dictate the sort
            order of query results.
        matching_strategy (Literal["last", "all"]): Defines the matching behavior for
            evaluating conditions. Options are "last" (only the last condition must match)
            or "all" (all conditions must match).
        attributes_to_search_on (tuple[str, ...]): A collection of attributes to target
            for searches. Defaults to searching across all attributes (`"*"`).
        search_options (SearchOptions): Additional configuration for search behavior, encapsulating
            more advanced settings.
    """
    offset: int = 0
    limit: int = 20
    filters: tuple[str, ...] = field(default_factory=tuple)
    sort: tuple[str, ...] = field(default_factory=tuple)
    matching_strategy: Literal["last", "all"] = "last"
    attributes_to_search_on: tuple[str, ...] = field(default_factory=lambda: ("*",))
    search_options: SearchOptions = field(default_factory=SearchOptions)

class Radius(NamedTuple):
    """
    Representation of a geographical radius with latitude, longitude, and radius size.

    This class is designed to define a geographical area using a center point
    specified by latitude and longitude, along with the radius size around the
    center point. It is useful in applications where defining a circular region
    is required (e.g., mapping, geofencing, etc.).

    Attributes:
        lat (float | str): The latitude value of the center point. Can be a float
            or a string representation of the value.
        lng (float | str): The longitude value of the center point. Can be a float
            or a string representation of the value.
        radius (int): The radius size around the center point, represented as an
            integer value, typically in meters.
    """

    lat: float | str
    lng: float | str
    radius: int


class BoundingBox(NamedTuple):
    """
    Represents a bounding box defined by its top-right and bottom-left coordinates.

    This class is useful for defining rectangular areas using coordinate points.
    Each coordinate can either be a float indicating precise location or a string
    representing a named placeholder. The bounding box can be utilized in
    geographical, graphical, or other positional contexts.

    Attributes:
        top_right (tuple[float | str, float | str]): A tuple representing the
            top-right corner of the bounding box as (x, y) coordinates.
        bottom_left (tuple[float | str, float | str]): A tuple representing the
            bottom-left corner of the bounding box as (x, y) coordinates.
    """

    top_right: tuple[float | str, float | str]
    bottom_left: tuple[float | str, float | str]


class Point(NamedTuple):
    """
    Represents a geographical point with latitude and longitude.

    This class is used to represent a specific point on the Earth's surface. It
    provides latitude and longitude values, which can be specified as either
    `float` or `str`. It is an immutable data structure, as it extends from
    `NamedTuple`. This makes it suitable for use in scenarios requiring hashable
    types, such as keys in dictionaries, or when ensuring the location data
    remains unchanged.

    Attributes:
        lat: Latitude of the point as a float or string.
        lng: Longitude of the point as a float or string.
    """

    lat: float | str
    lng: float | str


class IndexQuerySet:
    """
    Represents a queryset initialized with a specific model and manages its
    internal state for querying data in Meilisearch.

    This class allows building and applying filters, sorting criteria, and
    other parameters for querying data. It integrates with Meilisearch by
    fetching indices related to the model and provides utilities for modifying
    or cloning query states. It supports advanced operations such as slicing,
    field-level sorting, and filter expression construction.

    Attributes:
        model (Type[T]): The model associated with the queryset.
        index (Any): The Meilisearch index associated with the model.
        _state (QuerySetState): The state object that maintains the internal
            status and parameters of the queryset. Defaults to a new instance of
            QuerySetState if not provided.
    """

    def __init__(
            self,
            model: Type[T],
            state: Optional[QuerySetState] = None
    ):
        """
        Represents an initialization of a queryset with a specified model and optional state. It
        establishes a base model for queryset and creates or assigns a state object to manage
        queryset's internal state.

        Args:
            model: The model class that defines the structure and schema for the queryset.
            state: An optional state object of type QuerySetState. If not provided, a new
                QuerySetState instance is created.
        """
        self.model = model
        self.index = self._get_index(model)
        self._state = state or QuerySetState()


    @staticmethod
    @lru_cache(maxsize=10)
    def _get_index(model: Type[T]) -> Any:
        """
        Fetches the index for the given model from the Meilisearch client with caching to
        improve performance. The function utilizes an LRU cache to store up to a maximum
        of 10 recently accessed indices, which helps reduce repeated client calls for
        commonly used models.

        NOTE: Might want to make the cache size configurable in the future.

        Args:
            model (Type[T]): A type object representing the model whose index is to be
                fetched. The model is expected to have a '_meilisearch' dictionary
                attribute containing the "index_name" key.

        Returns:
            Any: The index obtained from the Meilisearch client.

        """
        return client.get_index(model._meilisearch["index_name"])

    def __repr__(self) -> str:
        return (f"<IndexQuerySet for {self.model.__name__} "
                f"filters={self._filters} sort={self._sort}>")

    def clone(self, **state_updates) -> Self:
        """Create a new instance with updated state."""
        new_state = replace(self._state, **state_updates)
        return self.__class__(self.model, new_state)

    def __getitem__(self, index: slice) -> Self:
        """Supports retrieving a sliced subset of the current object query set.

        Retrieves a clone of the current object with adjusted offset and limit
        based on the slice specified by the index. The slice should be defined
        with start and stop values to adjust the subset of data retrieved.

        Raises:
            TypeError: If the provided index is not a slice.

        Args:
            index (slice): A slice object defining the desired subset.
                The `start` indicates the offset to set for the query set,
                while the `stop` specifies the limit.

        Returns:
            Self: A new object with the adjusted offset and limit settings
            based on the given slice parameters.
        """
        if not isinstance(index, slice):
            raise TypeError("IndexQuerySet indices must be slices")

        return self.clone(
            offset=index.start if index.start is not None else self._state.offset,
            limit=index.stop if index.stop is not None else self._state.limit
        )

    def count(self) -> int:
        """Returns the number of documents in the index.

        Note: This method is not specific to the current queryset and will return the total number of documents in the index.
        """

        return self.index.get_stats().number_of_documents

    def _build_sort_expression(self, field: str) -> str:
        """
        Constructs a sort expression string based on a given field and its sorting
        direction.

        This method generates a string that indicates the field to be sorted and its
        direction (ascending or descending). If the field corresponds to a geoPoint,
        it includes a specific prefix in the returned string.

        Args:
            field (str): The field to construct the sort expression for. Fields
                prefixed with "-" are sorted in descending order, whereas others are
                sorted in ascending order. If the field represents geoPoint, it
                modifies the returned expression accordingly.

        Returns:
            str: A string representing the sort expression to be used.
        """
        geopoint = "_" if "geoPoint" in field else ""
        direction = "desc" if field.startswith("-") else "asc"
        clean_field = field[1:] if field.startswith("-") else field
        return f"{geopoint}{clean_field}:{direction}"

    def order_by(self, *fields: str) -> Self:
        """
        Creates and returns a new instance of the current object with updated sorting
        criteria based on the provided fields.

        Sort expressions are built for each field provided and used to define
        the sorting order for a new instance. The original object remains
        unchanged.

        Args:
            *fields (str): Arbitrary number of fields used to define the
                sorting order. Each field is processed to create a
                corresponding sort expression.

        Returns:
            Self: A new instance of the object with updated sorting
                configuration based on the provided fields.
        """
        return self.clone(
            sort=tuple(self._build_sort_expression(field) for field in fields)
        )

    def _build_filter_expression(
            self,
            field: str,
            operator: FilterOperator,
            value: Any
    ) -> str:
        """
        Builds a filter expression for a query based on field, operator, and value.

        This function generates a filter expression as a string depending on the
        specified filter operator and its associated value. It supports logical
        operations such as exact match, comparison (greater than, less than, etc.),
        inclusion, range checks, existence, and null checks. The resulting
        expression is typically used for constructing queries in a database or
        data-processing context.

        Args:
            field (str): The field name for the filter condition.
            operator (FilterOperator): The operator to apply for filtering
                (e.g., exact match, greater than, etc.).
            value (Any): The value to use with the filter operator, which should
                match the expected data type for the specified operator.

        Returns:
            str: A filter expression as a string based on the provided inputs.

        Raises:
            TypeError: If the type of the `value` is incompatible with the
                specified `operator`.
            ValueError: If an unknown `operator` is provided.
        """
        if operator == FilterOperator.EXACT:
            if value == "" or (isinstance(value, list) and not value) or value == {}:
                return f"{field} IS EMPTY"
            if value is None:
                return f"{field} IS NULL"
            return (f"{field} = '{value}'" if isinstance(value, str)
                    else f"{field} = {value}")

        if operator in (FilterOperator.GT, FilterOperator.GTE,
                        FilterOperator.LT, FilterOperator.LTE):
            if not isinstance(value, (int, float)):
                raise TypeError(f"Cannot compare {type(value)} with int or float")
            op_map = {
                FilterOperator.GT: ">",
                FilterOperator.GTE: ">=",
                FilterOperator.LT: "<",
                FilterOperator.LTE: "<="
            }
            return f"{field} {op_map[operator]} {value}"

        if operator == FilterOperator.IN:
            if not isinstance(value, (list, tuple, set)):
                raise TypeError(f"Cannot use IN with {type(value)}")
            return f"{field} IN {list(value)}"

        if operator == FilterOperator.RANGE:
            if not isinstance(value, (range, list, tuple)):
                raise TypeError(f"Cannot use RANGE with {type(value)}")
            start = value.start if isinstance(value, range) else value[0]
            stop = value.stop if isinstance(value, range) else value[1]
            return f"{field} {start} TO {stop}"

        if operator in (FilterOperator.EXISTS, FilterOperator.ISNULL):
            if not isinstance(value, bool):
                raise TypeError(f"Cannot compare {type(value)} with bool")
            if operator == FilterOperator.EXISTS:
                return f"{field} {'NOT ' if not value else ''}EXISTS"
            return f"{field} {'NOT ' if not value else ''}IS NULL"

        raise ValueError(f"Unknown operator: {operator}")

    def _parse_filter_key(self, key: str) -> tuple[str, FilterOperator]:
        """
        Parses a filter key into a field name and its associated filter operator.

        This method converts a filter key, which is a string, into a tuple
        containing a field name and a filter operator. If the filter key does
        not contain a valid operator, it defaults the operation to `EXACT`.
        The filter key format is expected to be a string with the field name and
        operator separated by double underscores (`__`). For example:
        `field__operator`.

        Args:
            key (str): The filter key containing the field name and the filter
                operator separated by double underscores.

        Returns:
            tuple[str, FilterOperator]: A tuple where the first element is the
                field name and the second element is the corresponding filter
                operator.

        Raises:
            ValueError: If the provided operator in the key is not a valid filter
                operator.
        """
        if "__" not in key:
            return key, FilterOperator.EXACT

        field, op = key.rsplit("__", 1)
        try:
            return field, FilterOperator(op)
        except ValueError:
            raise ValueError(f"Unknown filter operator: {op}")

    def filter(self, *geo_filters, **filters) -> Self:
        """
        Filters the current query based on geographical or attribute filters. This method allows
        you to refine query results using specific geographical filters (such as Radius or
        BoundingBox) or attribute-based filters.

        Geo filters can only be applied if the model supports geographical filtering. The provided
        geographical filters should be instances of either `Radius` or `BoundingBox`. If any other
        objects are provided, or if the model does not support geo filters, a `TypeError` will
        be raised.

        Attribute filters are specified as keyword arguments, with keys representing the filter
        field and optional operators, and values indicating the value(s) to filter on. Filters
        are built into expressions and added to the existing state.

        Args:
            *geo_filters (Radius or BoundingBox):
                Geographical filters to restrict the query results to a specific geographical
                area. This can include a `Radius` object to limit results within a circular
                region or a `BoundingBox` to specify a rectangular region.

            **filters (dict):
                Attribute-based filters to refine the query. Keys represent filter fields
                optionally with an operator, and values are the corresponding filter values.

        Raises:
            TypeError: If geographical filters are passed and the model does not support geo
                filtering, or if provided geo filters are not instances of `Radius` or
                `BoundingBox`.

        Returns:
            Self: A new query object with the applied filters.
        """
        new_filters = list(self._state.filters)

        # Handle geo filters
        for geo_filter in geo_filters:
            if not self.model._meilisearch["supports_geo"]:
                raise TypeError(
                    f"Model {self.model.__name__} does not support geo filters"
                )
            if not isinstance(geo_filter, (Radius, BoundingBox)):
                raise TypeError(
                    f"Geo filter must be Radius or BoundingBox, not {type(geo_filter)}"
                )

            if isinstance(geo_filter, Radius):
                new_filters.append(
                    f"_geoRadius({geo_filter.lat}, {geo_filter.lng}, "
                    f"{geo_filter.radius})"
                )
            else:  # BoundingBox
                new_filters.append(
                    f"_geoBoundingBox([{geo_filter.top_right[0]}, "
                    f"{geo_filter.top_right[1]}], [{geo_filter.bottom_left[0]}, "
                    f"{geo_filter.bottom_left[1]}])"
                )

        # Handle attribute filters
        for filter_key, value in filters.items():
            field, operator = self._parse_filter_key(filter_key)
            new_filters.append(
                self._build_filter_expression(field, operator, value)
            )

        return self.clone(filters=tuple(new_filters))

    def matching_strategy(self, strategy: Literal["last", "all"]) -> Self:
        """
        Configures the matching strategy for the object. The matching strategy
        determines the way matches are evaluated, allowing for different
        operational preferences such as matching the last occurrence or all
        occurrences of specified criteria.

        Args:
            strategy: A string literal specifying the matching strategy.
                Accepts "last" to match the last occurrence of criteria or
                "all" to match all occurrences.

        Returns:
            Self: A new instance of the object with the updated matching
            strategy applied.
        """
        return self.clone(matching_strategy=strategy)

    def attributes_to_search_on(self, *attributes: str) -> Self:
        """
        Allows specifying attributes to search on for the current object. This method clones
        the current instance and sets the provided attributes as the new value of
        `attributes_to_search_on`.

        Args:
            *attributes: Variable length argument list consisting of attribute names to
                search on.

        Returns:
            Self: A new instance with updated `attributes_to_search_on` attribute.
        """
        return self.clone(attributes_to_search_on=tuple(attributes))

    def with_search_options(self, **options) -> Self:
        """
        Updates the search options in the current object by replacing specific attributes.

        This method creates a new instance of the object with updated search options. Only
        attributes that already exist in the current search options state will be replaced
        with the provided keyword arguments. Attributes not present in the current state
        will be ignored.

        Args:
            **options: Arbitrary attributes that should potentially replace values in the
                current search options state.

        Returns:
            Self: A new instance of the class with the updated search options.
        """
        current_options = self._state.search_options
        new_options = {
            k: v for k, v in options.items()
            if hasattr(current_options, k)
        }
        return self.clone(
            search_options=replace(current_options, **new_options)
        )

    def _prepare_search_options(self) -> Dict[str, Any]:
        """
        Prepares a dictionary containing search options to be used in a search operation.

        This method processes and combines multiple aspects of the current search state,
        including offset, limit, filters, sorting options, matching strategies, and attributes
        to search on. Additionally, it includes any custom search options set in the
        `state.search_options` attribute. It ensures non-None values are added and may also
        process facets based on the provided facets' filters or exclusions.

        Returns:
            Dict[str, Any]: A dictionary representing the assembled search options to be
            used during the search operation.
        """
        state = self._state
        options = {
            "offset": state.offset,
            "limit": state.limit,
            "filter": list(state.filters),
            "sort": list(state.sort),
            "matchingStrategy": state.matching_strategy,
            "attributesToSearchOn": list(state.attributes_to_search_on),
        }

        # Add non-None search options
        search_options_dict = {
            k: v for k, v in state.search_options.__dict__.items()
            if v is not None
        }
        if search_options_dict.get('facets') == ['*']:
            exclude = search_options_dict.pop('facets_exclude', None)
            if exclude:
                search_options_dict['facets'] = [
                    f for f in self.model._meilisearch["filterable_fields"]
                    if f not in exclude
                ]
            else:
                search_options_dict['facets'] = (
                    self.model._meilisearch["filterable_fields"]
                )

        options.update(search_options_dict)
        return options

    def search(
            self,
            q: str = "",
            **kwargs
    ) -> "MeiliSearchResults[T]":
        """
        Performs a search operation on the associated MeiliSearch index with the provided query
        string and additional options. The search results from MeiliSearch are used to fetch the
        corresponding database objects in the base queryset, based on the primary key specified
        in the model's MeiliMeta.

        Args:
            q (str): The search query string to be executed against the MeiliSearch index. Defaults to an empty string.
            **kwargs: Additional search options or filters to customize the search behavior. These options
                will override or update the existing search configuration.

        Returns:
            MeiliSearchResults[T]: An instance containing the filtered database queryset and the raw results
            retrieved from the MeiliSearch index.

        """
        # Create new instance with updated search options if needed
        queryset = self
        if kwargs:
            queryset = self.with_search_options(**kwargs)

        results = self.index.search(q, queryset._prepare_search_options())

        id_field = getattr(self.model.MeiliMeta, "primary_key", "id")
        base_queryset = self.model.objects.filter(
            pk__in=[hit[id_field] for hit in results.get("hits", [])]
        )

        return MeiliSearchResults(base_queryset, results)


class MeiliSearchResults:
    """
    A lightweight wrapper that adds Meilisearch metadata to a Django QuerySet.
    Delegates all QuerySet operations to the underlying queryset while providing
    access to Meilisearch-specific data.
    """
    def __init__(self, queryset: "QuerySet[T]", meili_results: Dict[str, Any]):
        self.queryset = queryset
        # Meilisearch specific metadata
        self.facet_distribution = meili_results.get('facetDistribution')
        self.total_hits = meili_results.get('estimatedTotalHits')
        self.processing_time_ms = meili_results.get('processingTimeMs')
        self.query = meili_results.get('query')
        self.hits = meili_results.get('hits', [])

    def __getattr__(self, name):
        """
        Delegate all other attributes to the underlying queryset
        """
        return getattr(self.queryset, name)

    # iterator method
    def __iter__(self):
        """
        Allows the object to be iterable, delegating the iteration operation to the
        queryset's iteration mechanism.

        Returns:
            Iterator: An iterator object from the queryset to iterate over its elements.
        """
        return self.queryset.__iter__()

    def __len__(self):
        """
        Returns the number of elements in the queryset.

        This method is used to obtain the total count of elements present in
        the queryset associated with this object. It leverages the length
        functionality to measure the size of the dataset.

        Returns:
            int: The total number of elements in the queryset.
        """
        return len(self.queryset)

    def __getitem__(self, index):
        """
        Enables access to elements within the queryset by their index. This method
        provides an interface to retrieve specific elements, acting as a proxy to
        the underlying queryset's __getitem__ method.

        Args:
            index (int): The index position of the element to retrieve from the
                queryset.

        Returns:
            Any: The element at the specified index in the underlying queryset.

        """
        return self.queryset.__getitem__(index)

    def __contains__(self, item):
        """
        Checks whether the specified item exists in the queryset.

        This method allows for using the `in` operator to check if an item is present
        within the queryset. It delegates the containment check to the underlying
        queryset implementation.

        Args:
            item: The object to check for membership in the queryset.

        Returns:
            bool: True if the item exists in the queryset, otherwise False.
        """
        return self.queryset.__contains__(item)

    def __reversed__(self):
        """
        Reverses the order of elements in the queryset.

        This method provides the ability to iterate over the queryset in reverse
        order. It delegates the operation to the underlying queryset's __reversed__
        method.

        Returns:
            Iterable: An iterable representing the reversed queryset.
        """
        return self.queryset.__reversed__()

    def __or__(self, other: QuerySet) -> QuerySet:
        """
        Represents an operation that performs a logical OR between two QuerySet objects.

        Args:
            other (QuerySet): The QuerySet object to be combined with the current
                QuerySet using a logical OR operation.

        Returns:
            QuerySet: A new QuerySet resulting from the logical OR operation between
                the current QuerySet and the provided one.
        """
        return self.queryset.__or__(other)

    def __and__(self, other: QuerySet) -> QuerySet:
        """
        Performs a logical AND operation between two QuerySet objects.

        This method allows combining the current QuerySet object with another
        QuerySet object using the logical AND operator, producing a new QuerySet
        that contains objects common to both input QuerySets.

        Args:
            other (QuerySet): Another QuerySet object to perform the logical AND
                operation with.

        Returns:
            QuerySet: A new QuerySet containing objects that are present in both
            the current QuerySet and the provided one.
        """
        return self.queryset.__and__(other)

    def __repr__(self):
        """
        Overrides the default representation of the class to provide a customized
        string representation that replaces "QuerySet" with "MeiliSearchResults"
        in the original queryset's representation.

        Returns:
            str: A customized string representation of the queryset object, where
            "QuerySet" is replaced with "MeiliSearchResults".
        """
        return repr(self.queryset).replace("QuerySet", "MeiliSearchResults")

    def __str__(self):
        """
        Converts the `queryset` attribute into a formatted string representation.

        The `__str__` method customizes the string representation of the internal
        `queryset` attribute by replacing the "QuerySet" substring with
        "MeiliSearchResults". This method is typically invoked when the object is
        converted to a string (e.g., with `str` or when printed).

        Returns:
            str: A formatted string representation of the `queryset` attribute.
        """
        return str(self.queryset).replace("QuerySet", "MeiliSearchResults")

    def __eq__(self, other):
        """
        Checks equality between two objects of the same type based on their `queryset` attribute.

        This method overrides the equality operator to compare the `queryset` attribute
        between the current instance and another object. It ensures that two objects
        are considered equal if they share the same `queryset`.

        Args:
            other (object): The object to compare with the current instance.

        Returns:
            bool: True if `queryset` of both objects is equal, False otherwise.
        """
        return self.queryset == other.queryset

    def __ne__(self, other):
        """
        Compares the inequality between two objects based on their queryset attribute.

        The method overrides the default behavior for the "!=" operator, allowing
        comparison of objects based on the `queryset` attribute.

        Args:
            other: The object to compare with the current instance.

        Returns:
            bool: True if the `queryset` attribute of the current object is not equal
            to the `queryset` attribute of the other object, False otherwise.
        """
        return self.queryset != other.queryset

    def __lt__(self, other):
        """
        Compares the queryset of the current object with another object to determine
        if the former is less than the latter.

        This method is typically used to compare querysets of related objects in order
        to define a specific ordering or relationship between them.

        Args:
            other: The object to compare the queryset to. Must have a `queryset` attribute
                for the comparison to be valid.

        Returns:
            bool: True if the queryset of the current object is less than the queryset
            of the other object; otherwise, False.
        """
        return self.queryset < other.queryset

    def __le__(self, other):
        """
        Compares the current queryset with another queryset using the less than or equal comparison. This method allows for
        comparison of two querysets to determine if the current queryset contains fewer or equal elements compared to the
        other queryset.

        Args:
            other: Another instance with a queryset attribute to compare to the current queryset.

        Returns:
            bool: True if the current queryset is less than or equal to the other queryset, otherwise False.
        """
        return self.queryset <= other.queryset

    def __gt__(self, other):
        """
        Compares two objects based on their `queryset` attribute to determine if one object
        is greater than the other.

        Args:
            other: The object to compare against. Expected to have a `queryset` attribute.

        Returns:
            bool: True if the `queryset` value of the current object is greater than the
            `queryset` value of the `other` object. Otherwise, returns False.
        """
        return self.queryset > other.queryset

    def __ge__(self, other):
        """
        Implement the greater-than-or-equal-to (>=) comparison between the current
        object's queryset and another object's queryset.

        This method compares the queryset of the current object with the queryset
        of another object to determine if it is greater than or equal.

        Args:
            other: The object whose queryset is being compared to the queryset of
                the current object.

        Returns:
            bool: True if the current object's queryset is greater than or equal
            to the other object's queryset, otherwise False.
        """
        return self.queryset >= other.queryset