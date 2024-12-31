"""
querysets.py
Ian Kollipara <ian.kollipara@gmail.com>

This module contains the QuerySet classes for the Django MeiliSearch app.
"""

# Imports
from typing import TYPE_CHECKING, Literal, NamedTuple, Self, Type, Dict, Any, Optional, TypeVar, Generic
from django.db.models import QuerySet

from ._client import client

if TYPE_CHECKING:
    from .models import IndexMixin


class Radius(NamedTuple):
    """A radius for a geosearch query."""

    lat: float | str
    lng: float | str
    radius: int


class BoundingBox(NamedTuple):
    """A bounding box for a geosearch query."""

    top_right: tuple[float | str, float | str]
    bottom_left: tuple[float | str, float | str]


class Point(NamedTuple):
    """A point for a geosearch query."""

    lat: float | str
    lng: float | str


class IndexQuerySet:
    """QuerySet for a MeiliSearch index.

    This class provides a way to interact with a MeiliSearch index for a given model.
    The queryset mimics the Django QuerySet API and provides methods to filter, sort, and search the index.
    """

    def __init__(self, model: Type["IndexMixin"]):
        self.model = model
        self.index = client.get_index(model._meilisearch["index_name"])
        self.__offset = 0
        self.__limit = 20
        self.__filters: list[str] = []
        self.__sort: list[str] = []
        self.__matching_strategy: Literal["last", "all"] = "last"
        self.__attributes_to_search_on: list[str] = ["*"]

    def __repr__(self):
        return f"<IndexQuerySet for {self.model.__name__}>"

    def __str__(self):
        return f"IndexQuerySet for {self.model.__name__}"

    def __getitem__(self, index):
        if isinstance(index, slice):
            self.__offset = index.start
            self.__limit = index.stop
            return self
        else:
            raise TypeError("IndexQuerySet indices must be slices")

    def count(self) -> int:
        """Returns the number of documents in the index.

        Note: This method is not specific to the current queryset and will return the total number of documents in the index.
        """

        return self.index.get_stats().number_of_documents

    def order_by(self, *fields: str):
        """Orders the queryset by the given fields.

        This mimics the Django QuerySet API and allows for ordering by multiple fields.
        The fields can be prefixed with "-" to indicate descending order.

        For example:
        ```python
        Model.meilisearch.order_by("field1", "-field2")
        ```

        For geosearch, the special geoPoint construct can be used.

        For example:
        ```python
        Model.meilisearch.order_by("geoPoint(lat, lng)")
        Model.meilisearch.order_by("-geoPoint(lat, lng)")
        ```
        """

        for field in fields:
            geopoint = "_" if "geoPoint" in field else ""
            if field.startswith("-"):
                self.__sort.append(f"{geopoint}{field[1:]}:desc")
            else:
                self.__sort.append(f"{geopoint}{field}:asc")
        return self

    def filter(self, *geo_filters, **filters) -> Self:
        """Filters the queryset by the given filters.

        This set of filtering mimics the Django QuerySet API and allows for filtering by multiple fields.
        The currently implemented filters are:
        - exact: Filters for an exact match.
        - gte: Filters for greater than or equal to.
        - gt: Filters for greater than.
        - lte: Filters for less than or equal to.
        - lt: Filters for less than.
        - in: Filters for a value in a list.
        - range: Filters for a value in a range.
        - exists: Filters for the existence of a field.
        - isnull: Filters for the nullness of a field.

        For example:
        ```python
        Model.meilisearch.filter(field1__exact="value", field2__gte=10, field3__in=[1, 2, 3])
        ```

        For geosearch, the Radius and BoundingBox classes can be used, and should be passed as unnamed arguments.
        If the model does not support geosearch, a TypeError will be raised.
        If the provided positional arguments are not of type Radius or BoundingBox, a TypeError will be raised.

        For example:
        ```python
        Model.meilisearch.filter(Radius(lat, lng, radius))
        Model.meilisearch.filter(BoundingBox(top_right, bottom_left))
        ```
        """

        for geo_filter in geo_filters:
            if not self.model._meilisearch["supports_geo"]:
                raise TypeError(
                    f"Model {self.model.__name__} does not support geo filters"
                )
            if not isinstance(geo_filter, (Radius, BoundingBox)):
                raise TypeError(
                    f"Unnamed Argument must be of type Radius or BoundingBox, not {type(geo_filter)}"
                )
            if isinstance(geo_filter, Radius):
                self.__filters.append(
                    f"_geoRadius({geo_filter.lat}, {geo_filter.lng}, {geo_filter.radius})"
                )
            elif isinstance(geo_filter, BoundingBox):
                self.__filters.append(
                    f"_geoBoundingBox([{geo_filter.top_right[0]}, {geo_filter.top_right[1]}], [{geo_filter.bottom_left[0]}, {geo_filter.bottom_left[1]}])"
                )
        for filter, value in filters.items():
            if "__" not in filter or "__exact" in filter:
                if (
                    value == ""
                    or (isinstance(value, list) and len(value) == 0)
                    or value == {}
                ):
                    self.__filters.append(f"{filter.split('__')[0]} IS EMPTY")
                elif value is None:
                    self.__filters.append(f"{filter.split('__')[0]} IS NULL")
                else:
                    self.__filters.append(
                        f"{filter.split('__')[0]} = '{value}'"
                        if isinstance(value, str)
                        else f"{filter.split('__')[0]} = {value}"
                    )
            elif "__gte" in filter:
                if not isinstance(value, (int, float)):
                    raise TypeError(f"Cannot compare {type(value)} with int or float")
                self.__filters.append(f"{filter.split('__')[0]} >= {value}")
            elif "__gt" in filter:
                if not isinstance(value, (int, float)):
                    raise TypeError(f"Cannot compare {type(value)} with int or float")
                self.__filters.append(f"{filter.split('__')[0]} > {value}")
            elif "__lte" in filter:
                if not isinstance(value, (int, float)):
                    raise TypeError(f"Cannot compare {type(value)} with int or float")
                self.__filters.append(f"{filter.split('__')[0]} <= {value}")
            elif "__lt" in filter:
                if not isinstance(value, (int, float)):
                    raise TypeError(f"Cannot compare {type(value)} with int or float")
                self.__filters.append(f"{filter.split('__')[0]} < {value}")
            elif "__in" in filter:
                if not isinstance(value, list):
                    raise TypeError(f"Cannot compare {type(value)} with list")
                self.__filters.append(f"{filter.split('__')[0]} IN {value}")
            elif "__range" in filter:
                if not isinstance(value, (range, list, tuple)):
                    raise TypeError(
                        f"Cannot compare {type(value)} with range, list or tuple"
                    )
                self.__filters.append(
                    f"{filter.split('__')[0]} {value[0]} TO {value[1]}"
                    if not isinstance(value, range)
                    else f"{filter.split('__')[0]} {value.start} TO {value.stop}"
                )
            elif "__exists" in filter:
                if not isinstance(value, bool):
                    raise TypeError(f"Cannot compare {type(value)} with bool")
                self.__filters.append(
                    f"{filter.split('__')[0]} {'NOT ' if not value else ''}EXISTS"
                )
            elif "__isnull" in filter:
                if not isinstance(value, bool):
                    raise TypeError(f"Cannot compare {type(value)} with bool")
                self.__filters.append(
                    f"{filter.split('__')[0]} {'NOT ' if not value else ''}IS NULL"
                )

        return self

    def matching_strategy(self, strategy: Literal["last", "all"]):
        """Sets the matching strategy for the search.

        The matching strategy can be either "last" or "all".
        """

        self.__matching_strategy = strategy
        return self

    def attributes_to_search_on(self, *attributes):
        """Sets the attributes to search on.

        This method allows for setting the attributes to search on for the search query.

        For example:
        ```python
        Model.meilisearch.attributes_to_search_on("title", "body")
        ```
        """

        self.__attributes_to_search_on.append(*attributes)
        return self

    def _get_search_options(self, options: Dict[str, Any] = None) -> Dict[str, Any]:
        if options is None:
            options = {}
        if "facets" in options:
            facets = options.get("facets", [])
            if len(facets) == 1 and facets[0] == "*":
                options["facets"] = self.model._meilisearch["filterable_fields"]
        return options

    def search(self, q: str = "", options: Dict[str, Any] = None) -> "MeiliSearchResults":
        """Searches the index for the given query.

        This method searches the index for the given query and returns the results as a MeiliSearchResults,
        which includes both the Django queryset results and Meilisearch metadata like faceting.

        For example:
        ```python
        results = Model.meilisearch.search("Hello World", options={"facets": ["category"]})
        # Access the queryset results
        for obj in results:
            print(obj)
        # Access faceting data
        print(results.facet_distribution)
        ```
        """
        options = self._get_search_options(options)

        results = self.index.search(
            q,
            {
                "offset": self.__offset,
                "limit": self.__limit,
                "filter": self.__filters,
                "sort": self.__sort,
                "matchingStrategy": self.__matching_strategy,
                "attributesToSearchOn": self.__attributes_to_search_on,
                **options,
            },
        )
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
    def __init__(self, queryset: QuerySet["IndexMixin"], meili_results: Dict[str, Any]):
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
        return self.queryset.__iter__()

    def __len__(self):
        return len(self.queryset)

    def __getitem__(self, index):
        return self.queryset.__getitem__(index)

    def __contains__(self, item):
        return self.queryset.__contains__(item)

    def __reversed__(self):
        return self.queryset.__reversed__()

    def __or__(self, other: QuerySet) -> QuerySet:
        return self.queryset.__or__(other)

    def __and__(self, other: QuerySet) -> QuerySet:
        return self.queryset.__and__(other)

    def __repr__(self):
        return repr(self.queryset).replace("QuerySet", "MeiliSearchResults")

    def __str__(self):
        return str(self.queryset).replace("QuerySet", "MeiliSearchResults")

    def __eq__(self, other):
        return self.queryset == other.queryset

    def __ne__(self, other):
        return self.queryset != other.queryset

    def __lt__(self, other):
        return self.queryset < other.queryset

    def __le__(self, other):
        return self.queryset <= other.queryset

    def __gt__(self, other):
        return self.queryset > other.queryset

    def __ge__(self, other):
        return self.queryset >= other.queryset