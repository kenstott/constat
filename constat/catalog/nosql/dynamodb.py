"""AWS DynamoDB connector for key-value/document database support."""

from typing import Optional, Any

from .base import NoSQLConnector, NoSQLType, CollectionMetadata, FieldInfo


class DynamoDBConnector(NoSQLConnector):
    """Connector for AWS DynamoDB.

    Usage:
        # Using default credentials (AWS CLI, environment, IAM role)
        connector = DynamoDBConnector(
            region="us-east-1",
            name="dynamodb_main",
        )

        # Using explicit credentials
        connector = DynamoDBConnector(
            region="us-east-1",
            aws_access_key_id="AKIA...",
            aws_secret_access_key="...",
            name="dynamodb_main",
        )

        # Local DynamoDB (for testing)
        connector = DynamoDBConnector(
            endpoint_url="http://localhost:8000",
            name="dynamodb_local",
        )

        connector.connect()
        tables = connector.get_collections()
        schema = connector.get_collection_schema("users")
    """

    # DynamoDB type mapping
    TYPE_MAP = {
        "S": "string",
        "N": "number",
        "B": "binary",
        "SS": "string_set",
        "NS": "number_set",
        "BS": "binary_set",
        "M": "map",
        "L": "list",
        "NULL": "null",
        "BOOL": "boolean",
    }

    def __init__(
        self,
        region: Optional[str] = None,
        name: Optional[str] = None,
        description: str = "",
        endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        profile_name: Optional[str] = None,
        sample_size: int = 100,
    ):
        """
        Initialize DynamoDB connector.

        Args:
            region: AWS region (e.g., "us-east-1")
            name: Friendly name for this connection
            description: Description of the database
            endpoint_url: Custom endpoint (for local DynamoDB)
            aws_access_key_id: AWS access key
            aws_secret_access_key: AWS secret key
            aws_session_token: AWS session token (for temporary credentials)
            profile_name: AWS profile name from credentials file
            sample_size: Number of items to sample for schema inference
        """
        super().__init__(name=name or "dynamodb", description=description)
        self.region = region
        self.endpoint_url = endpoint_url
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.profile_name = profile_name
        self.sample_size = sample_size
        self._client = None
        self._resource = None

    @property
    def nosql_type(self) -> NoSQLType:
        return NoSQLType.KEY_VALUE

    def connect(self) -> None:
        """Connect to DynamoDB."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "DynamoDB connector requires boto3. "
                "Install with: pip install boto3"
            )

        session_kwargs = {}
        if self.profile_name:
            session_kwargs["profile_name"] = self.profile_name
        if self.region:
            session_kwargs["region_name"] = self.region

        session = boto3.Session(**session_kwargs)

        client_kwargs = {}
        if self.endpoint_url:
            client_kwargs["endpoint_url"] = self.endpoint_url
        if self.aws_access_key_id:
            client_kwargs["aws_access_key_id"] = self.aws_access_key_id
            client_kwargs["aws_secret_access_key"] = self.aws_secret_access_key
        if self.aws_session_token:
            client_kwargs["aws_session_token"] = self.aws_session_token

        self._client = session.client("dynamodb", **client_kwargs)
        self._resource = session.resource("dynamodb", **client_kwargs)
        self._connected = True

    def disconnect(self) -> None:
        """Disconnect from DynamoDB."""
        self._client = None
        self._resource = None
        self._connected = False

    def get_collections(self) -> list[str]:
        """List all tables."""
        if not self._client:
            raise RuntimeError("Not connected to DynamoDB")

        tables = []
        paginator = self._client.get_paginator("list_tables")
        for page in paginator.paginate():
            tables.extend(page["TableNames"])
        return tables

    def get_collection_schema(self, collection: str) -> CollectionMetadata:
        """Get schema for a table."""
        if collection in self._metadata_cache:
            return self._metadata_cache[collection]

        if not self._client:
            raise RuntimeError("Not connected to DynamoDB")

        # Get table description
        response = self._client.describe_table(TableName=collection)
        table_desc = response["Table"]

        # Get key schema
        key_schema = {
            item["AttributeName"]: item["KeyType"]
            for item in table_desc["KeySchema"]
        }

        # Get attribute definitions
        attr_defs = {
            item["AttributeName"]: self.TYPE_MAP.get(item["AttributeType"], "unknown")
            for item in table_desc["AttributeDefinitions"]
        }

        # Build fields from key schema
        fields = []
        partition_key = None
        sort_key = None

        for attr_name, key_type in key_schema.items():
            field = FieldInfo(
                name=attr_name,
                data_type=attr_defs.get(attr_name, "unknown"),
                nullable=False,  # Key attributes are required
                is_indexed=True,
                is_unique=key_type == "HASH",
            )
            fields.append(field)

            if key_type == "HASH":
                partition_key = attr_name
            elif key_type == "RANGE":
                sort_key = attr_name

        # Sample items to discover non-key attributes
        sampled_fields = self._sample_table_schema(collection)
        for field in sampled_fields:
            if field.name not in key_schema:
                fields.append(field)

        # Get GSI and LSI info
        indexes = []
        for gsi in table_desc.get("GlobalSecondaryIndexes", []):
            indexes.append(f"GSI:{gsi['IndexName']}")
        for lsi in table_desc.get("LocalSecondaryIndexes", []):
            indexes.append(f"LSI:{lsi['IndexName']}")

        # Get item count (approximate)
        item_count = table_desc.get("ItemCount", 0)

        metadata = CollectionMetadata(
            name=collection,
            database="dynamodb",
            nosql_type=self.nosql_type,
            fields=fields,
            document_count=item_count,
            size_bytes=table_desc.get("TableSizeBytes", 0),
            indexes=indexes,
            partition_key=partition_key,
            clustering_keys=[sort_key] if sort_key else [],
        )

        self._metadata_cache[collection] = metadata
        return metadata

    def _sample_table_schema(self, collection: str) -> list[FieldInfo]:
        """Sample table to discover additional attributes."""
        if not self._resource:
            return []

        table = self._resource.Table(collection)
        response = table.scan(Limit=self.sample_size)
        items = response.get("Items", [])

        # Collect all attribute names and types
        field_values: dict[str, list[Any]] = {}
        for item in items:
            for key, value in item.items():
                if key not in field_values:
                    field_values[key] = []
                field_values[key].append(value)

        fields = []
        for field_name, values in field_values.items():
            fields.append(FieldInfo(
                name=field_name,
                data_type=self.infer_field_type(values),
                nullable=len(values) < len(items),
                sample_values=values[:3],
            ))

        return fields

    def query(
        self,
        collection: str,
        query: dict,
        limit: int = 100,
    ) -> list[dict]:
        """Query a table using key conditions.

        Args:
            collection: Table name
            query: Key condition expression as dict
            limit: Maximum items to return

        Returns:
            List of items
        """
        if not self._resource:
            raise RuntimeError("Not connected to DynamoDB")

        from boto3.dynamodb.conditions import Key, Attr

        table = self._resource.Table(collection)

        # Build key condition expression
        key_condition = None
        filter_expression = None

        for key, value in query.items():
            if isinstance(value, dict):
                # Handle operators
                for op, val in value.items():
                    condition = None
                    if op == "$eq":
                        condition = Key(key).eq(val)
                    elif op == "$gt":
                        condition = Key(key).gt(val)
                    elif op == "$gte":
                        condition = Key(key).gte(val)
                    elif op == "$lt":
                        condition = Key(key).lt(val)
                    elif op == "$lte":
                        condition = Key(key).lte(val)
                    elif op == "$begins_with":
                        condition = Key(key).begins_with(val)
                    elif op == "$between":
                        condition = Key(key).between(val[0], val[1])

                    if condition:
                        if key_condition is None:
                            key_condition = condition
                        else:
                            key_condition = key_condition & condition
            else:
                condition = Key(key).eq(value)
                if key_condition is None:
                    key_condition = condition
                else:
                    key_condition = key_condition & condition

        if key_condition:
            response = table.query(
                KeyConditionExpression=key_condition,
                Limit=limit,
            )
        else:
            # No key condition - use scan
            response = table.scan(Limit=limit)

        return response.get("Items", [])

    def scan(
        self,
        collection: str,
        filter_expr: Optional[dict] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Scan a table with optional filter.

        Args:
            collection: Table name
            filter_expr: Filter expression as dict
            limit: Maximum items to return

        Returns:
            List of items
        """
        if not self._resource:
            raise RuntimeError("Not connected to DynamoDB")

        from boto3.dynamodb.conditions import Attr

        table = self._resource.Table(collection)

        scan_kwargs = {"Limit": limit}

        if filter_expr:
            filter_condition = None
            for key, value in filter_expr.items():
                if isinstance(value, dict):
                    for op, val in value.items():
                        if op == "$eq":
                            condition = Attr(key).eq(val)
                        elif op == "$ne":
                            condition = Attr(key).ne(val)
                        elif op == "$gt":
                            condition = Attr(key).gt(val)
                        elif op == "$gte":
                            condition = Attr(key).gte(val)
                        elif op == "$lt":
                            condition = Attr(key).lt(val)
                        elif op == "$lte":
                            condition = Attr(key).lte(val)
                        elif op == "$exists":
                            condition = Attr(key).exists() if val else Attr(key).not_exists()
                        elif op == "$contains":
                            condition = Attr(key).contains(val)
                        else:
                            continue

                        if filter_condition is None:
                            filter_condition = condition
                        else:
                            filter_condition = filter_condition & condition
                else:
                    condition = Attr(key).eq(value)
                    if filter_condition is None:
                        filter_condition = condition
                    else:
                        filter_condition = filter_condition & condition

            if filter_condition:
                scan_kwargs["FilterExpression"] = filter_condition

        response = table.scan(**scan_kwargs)
        return response.get("Items", [])

    def insert(self, collection: str, documents: list[dict]) -> list[str]:
        """Put items into a table.

        Args:
            collection: Table name
            documents: Items to insert

        Returns:
            List of item identifiers (partition key values)
        """
        if not self._resource:
            raise RuntimeError("Not connected to DynamoDB")

        table = self._resource.Table(collection)
        ids = []

        with table.batch_writer() as batch:
            for doc in documents:
                batch.put_item(Item=doc)
                # Return the first key field as identifier
                for key in doc:
                    ids.append(str(doc[key]))
                    break

        return ids

    def update(
        self,
        collection: str,
        key: dict,
        updates: dict,
    ) -> dict:
        """Update an item.

        Args:
            collection: Table name
            key: Primary key of item to update
            updates: Attribute updates

        Returns:
            Updated item
        """
        if not self._resource:
            raise RuntimeError("Not connected to DynamoDB")

        table = self._resource.Table(collection)

        # Build update expression
        update_expr_parts = []
        expr_attr_values = {}
        expr_attr_names = {}

        for i, (attr, value) in enumerate(updates.items()):
            placeholder_name = f"#attr{i}"
            placeholder_value = f":val{i}"
            update_expr_parts.append(f"{placeholder_name} = {placeholder_value}")
            expr_attr_names[placeholder_name] = attr
            expr_attr_values[placeholder_value] = value

        update_expr = "SET " + ", ".join(update_expr_parts)

        response = table.update_item(
            Key=key,
            UpdateExpression=update_expr,
            ExpressionAttributeNames=expr_attr_names,
            ExpressionAttributeValues=expr_attr_values,
            ReturnValues="ALL_NEW",
        )

        return response.get("Attributes", {})

    def delete(self, collection: str, query: dict) -> int:
        """Delete items from a table.

        Args:
            collection: Table name
            query: Primary key of item(s) to delete

        Returns:
            Number of deleted items
        """
        if not self._resource:
            raise RuntimeError("Not connected to DynamoDB")

        table = self._resource.Table(collection)
        table.delete_item(Key=query)
        return 1

    def create_table(
        self,
        collection: str,
        partition_key: str,
        partition_key_type: str = "S",
        sort_key: Optional[str] = None,
        sort_key_type: str = "S",
        billing_mode: str = "PAY_PER_REQUEST",
    ) -> None:
        """Create a new table.

        Args:
            collection: Table name
            partition_key: Partition key attribute name
            partition_key_type: S (string), N (number), or B (binary)
            sort_key: Optional sort key attribute name
            sort_key_type: S, N, or B
            billing_mode: PAY_PER_REQUEST or PROVISIONED
        """
        if not self._client:
            raise RuntimeError("Not connected to DynamoDB")

        key_schema = [
            {"AttributeName": partition_key, "KeyType": "HASH"},
        ]
        attribute_defs = [
            {"AttributeName": partition_key, "AttributeType": partition_key_type},
        ]

        if sort_key:
            key_schema.append({"AttributeName": sort_key, "KeyType": "RANGE"})
            attribute_defs.append({"AttributeName": sort_key, "AttributeType": sort_key_type})

        self._client.create_table(
            TableName=collection,
            KeySchema=key_schema,
            AttributeDefinitions=attribute_defs,
            BillingMode=billing_mode,
        )

    def delete_table(self, collection: str) -> None:
        """Delete a table.

        Args:
            collection: Table name
        """
        if not self._client:
            raise RuntimeError("Not connected to DynamoDB")

        self._client.delete_table(TableName=collection)
