"""
Test helper utilities and common test functions.

This module provides shared utilities, test data factories, and helper functions
used across different test modules.
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from unittest.mock import MagicMock, patch
import pytest
from faker import Faker

# Initialize Faker
fake = Faker()


class TestDataFactory:
    """Factory for generating test data."""

    @staticmethod
    def create_user(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a fake user dictionary."""
        user = {
            "id": fake.uuid4(),
            "name": fake.name(),
            "email": fake.email(),
            "username": fake.user_name(),
            "password": fake.password(length=12),
            "phone": fake.phone_number(),
            "address": fake.address(),
            "city": fake.city(),
            "country": fake.country(),
            "postal_code": fake.postcode(),
            "date_of_birth": fake.date_of_birth(minimum_age=18, maximum_age=80),
            "created_at": fake.date_time_this_year(),
            "updated_at": fake.date_time_this_year(),
            "is_active": True,
            "is_verified": fake.boolean(chance_of_getting_true=80),
            "role": fake.random_element(["user", "admin", "moderator"]),
            "preferences": {
                "theme": fake.random_element(["light", "dark"]),
                "notifications": fake.boolean(),
                "language": fake.random_element(["en", "es", "fr", "de"])
            }
        }
        if overrides:
            user.update(overrides)
        return user

    @staticmethod
    def create_company(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a fake company dictionary."""
        company = {
            "id": fake.uuid4(),
            "name": fake.company(),
            "domain": fake.domain_name(),
            "description": fake.text(max_nb_chars=200),
            "industry": fake.random_element([
                "Technology", "Healthcare", "Finance", "Education",
                "Manufacturing", "Retail", "Consulting", "Media"
            ]),
            "size": fake.random_element(["1-10", "11-50", "51-200", "201-1000", "1000+"]),
            "founded_year": fake.year(),
            "website": fake.url(),
            "address": fake.address(),
            "phone": fake.phone_number(),
            "email": fake.company_email(),
            "created_at": fake.date_time_this_year(),
            "is_active": True
        }
        if overrides:
            company.update(overrides)
        return company

    @staticmethod
    def create_product(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a fake product dictionary."""
        from decimal import Decimal
        product = {
            "id": fake.uuid4(),
            "name": fake.word().title() + " " + fake.word().title(),
            "description": fake.text(max_nb_chars=300),
            "sku": fake.ean13(),
            "price": Decimal(str(fake.pydecimal(left_digits=3, right_digits=2, positive=True))),
            "cost": Decimal(str(fake.pydecimal(left_digits=2, right_digits=2, positive=True))),
            "category": fake.random_element([
                "Electronics", "Clothing", "Books", "Home", "Sports",
                "Beauty", "Automotive", "Garden", "Toys", "Food"
            ]),
            "brand": fake.company(),
            "weight": fake.pydecimal(left_digits=2, right_digits=2, positive=True),
            "dimensions": {
                "length": fake.pydecimal(left_digits=2, right_digits=1, positive=True),
                "width": fake.pydecimal(left_digits=2, right_digits=1, positive=True),
                "height": fake.pydecimal(left_digits=2, right_digits=1, positive=True)
            },
            "inventory_count": fake.random_int(min=0, max=1000),
            "is_active": fake.boolean(chance_of_getting_true=90),
            "created_at": fake.date_time_this_year(),
            "tags": [fake.word() for _ in range(fake.random_int(min=1, max=5))]
        }
        if overrides:
            product.update(overrides)
        return product

    @staticmethod
    def create_order(user_id: str, product_ids: List[str], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a fake order dictionary."""
        from decimal import Decimal
        order_items = []
        total_amount = Decimal('0.00')

        for product_id in product_ids:
            quantity = fake.random_int(min=1, max=5)
            unit_price = Decimal(str(fake.pydecimal(left_digits=2, right_digits=2, positive=True)))
            item_total = unit_price * quantity
            total_amount += item_total

            order_items.append({
                "product_id": product_id,
                "quantity": quantity,
                "unit_price": unit_price,
                "total_price": item_total
            })

        order = {
            "id": fake.uuid4(),
            "user_id": user_id,
            "order_number": fake.ean8(),
            "status": fake.random_element(["pending", "processing", "shipped", "delivered", "cancelled"]),
            "order_items": order_items,
            "subtotal": total_amount,
            "tax_amount": total_amount * Decimal('0.08'),  # 8% tax
            "shipping_amount": Decimal(str(fake.pydecimal(left_digits=2, right_digits=2, positive=True))),
            "total_amount": total_amount * Decimal('1.08') + Decimal(str(fake.pydecimal(left_digits=2, right_digits=2, positive=True))),
            "shipping_address": {
                "street": fake.street_address(),
                "city": fake.city(),
                "state": fake.state_abbr(),
                "postal_code": fake.postcode(),
                "country": fake.country_code()
            },
            "billing_address": {
                "street": fake.street_address(),
                "city": fake.city(),
                "state": fake.state_abbr(),
                "postal_code": fake.postcode(),
                "country": fake.country_code()
            },
            "payment_method": fake.random_element(["credit_card", "paypal", "bank_transfer"]),
            "created_at": fake.date_time_this_year(),
            "updated_at": fake.date_time_this_year()
        }

        if overrides:
            order.update(overrides)
            # Recalculate totals if items were overridden
            if "order_items" in overrides:
                new_total = sum(item["total_price"] for item in order["order_items"])
                order["subtotal"] = new_total
                order["total_amount"] = new_total * Decimal('1.08') + order["shipping_amount"]

        return order

    @staticmethod
    def create_api_response(status_code: int = 200, data: Optional[Any] = None, error: Optional[str] = None) -> Dict[str, Any]:
        """Create a fake API response."""
        response = {
            "status_code": status_code,
            "headers": {
                "Content-Type": "application/json",
                "X-Request-ID": fake.uuid4()
            }
        }

        if status_code >= 200 and status_code < 300:
            response["data"] = data or {"message": "Success"}
        else:
            response["error"] = error or f"Error {status_code}"

        return response


class MockFactory:
    """Factory for creating mock objects."""

    @staticmethod
    def create_mock_database_session():
        """Create a mock database session."""
        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = MagicMock()
        mock_session.refresh = MagicMock()
        mock_session.query = MagicMock()
        mock_session.filter = MagicMock()
        mock_session.first = MagicMock()
        mock_session.all = MagicMock()
        mock_session.delete = MagicMock()
        return mock_session

    @staticmethod
    def create_mock_redis_client():
        """Create a mock Redis client."""
        mock_redis = MagicMock()
        mock_redis.get = MagicMock(return_value=None)
        mock_redis.set = MagicMock(return_value=True)
        mock_redis.delete = MagicMock(return_value=True)
        mock_redis.exists = MagicMock(return_value=False)
        mock_redis.expire = MagicMock(return_value=True)
        return mock_redis

    @staticmethod
    def create_mock_http_client():
        """Create a mock HTTP client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={"status": "success"})
        mock_response.text = '{"status": "success"}'
        mock_client.get = MagicMock(return_value=mock_response)
        mock_client.post = MagicMock(return_value=mock_response)
        mock_client.put = MagicMock(return_value=mock_response)
        mock_client.delete = MagicMock(return_value=mock_response)
        return mock_client

    @staticmethod
    def create_mock_file_system():
        """Create a mock file system."""
        mock_fs = MagicMock()
        mock_fs.exists = MagicMock(return_value=True)
        mock_fs.isfile = MagicMock(return_value=True)
        mock_fs.isdir = MagicMock(return_value=False)
        mock_fs.read_text = MagicMock(return_value="file content")
        mock_fs.write_text = MagicMock()
        mock_fs.mkdir = MagicMock()
        return mock_fs


class AssertionHelpers:
    """Helper methods for common assertions."""

    @staticmethod
    def assert_dict_contains_subset(subset: Dict[str, Any], superset: Dict[str, Any]):
        """Assert that all keys and values in subset are present in superset."""
        for key, value in subset.items():
            assert key in superset, f"Key '{key}' not found in superset"
            if isinstance(value, dict):
                AssertionHelpers.assert_dict_contains_subset(value, superset[key])
            else:
                assert superset[key] == value, f"Value for key '{key}' does not match: expected {value}, got {superset[key]}"

    @staticmethod
    def assert_list_contains_items(items: List[Any], container: List[Any]):
        """Assert that all items are present in the container list."""
        for item in items:
            assert item in container, f"Item '{item}' not found in container"

    @staticmethod
    def assert_valid_email(email: str):
        """Assert that a string is a valid email format."""
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        assert re.match(email_pattern, email), f"Invalid email format: {email}"

    @staticmethod
    def assert_valid_uuid(uuid_string: str):
        """Assert that a string is a valid UUID."""
        import uuid
        try:
            uuid.UUID(uuid_string)
        except ValueError:
            pytest.fail(f"Invalid UUID format: {uuid_string}")

    @staticmethod
    def assert_response_status(response, expected_status: int):
        """Assert that an HTTP response has the expected status code."""
        assert response.status_code == expected_status, \
            f"Expected status {expected_status}, got {response.status_code}"

    @staticmethod
    def assert_response_contains_data(response, data_keys: List[str]):
        """Assert that a response contains specific data keys."""
        response_data = response.json()
        for key in data_keys:
            assert key in response_data, f"Key '{key}' not found in response data"


class FileSystemHelpers:
    """Helper methods for file system operations in tests."""

    @staticmethod
    def create_temp_file(content: str = "", suffix: str = ".txt") -> Path:
        """Create a temporary file with optional content."""
        fd, path = tempfile.mkstemp(suffix=suffix)
        temp_path = Path(path)
        with temp_path.open('w') as f:
            f.write(content)
        os.close(fd)
        return temp_path

    @staticmethod
    def create_temp_dir() -> Path:
        """Create a temporary directory."""
        return Path(tempfile.mkdtemp())

    @staticmethod
    def create_test_file_structure(base_dir: Path, structure: Dict[str, Any]) -> None:
        """Create a nested file/directory structure for testing."""
        for name, content in structure.items():
            path = base_dir / name
            if isinstance(content, dict):
                # It's a directory
                path.mkdir(parents=True, exist_ok=True)
                FileSystemHelpers.create_test_file_structure(path, content)
            else:
                # It's a file
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('w') as f:
                    f.write(content)

    @staticmethod
    def read_json_file(file_path: Path) -> Dict[str, Any]:
        """Read and parse a JSON file."""
        with file_path.open('r') as f:
            return json.load(f)

    @staticmethod
    def write_json_file(file_path: Path, data: Dict[str, Any]) -> None:
        """Write data to a JSON file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open('w') as f:
            json.dump(data, f, indent=2)


class DatabaseHelpers:
    """Helper methods for database operations in tests."""

    @staticmethod
    def setup_test_database():
        """Set up a test database with initial data."""
        # This would be implemented based on your specific database setup
        pass

    @staticmethod
    def teardown_test_database():
        """Clean up test database."""
        # This would be implemented based on your specific database setup
        pass

    @staticmethod
    def create_test_data(db_session, data_type: str, count: int = 1) -> List[Dict[str, Any]]:
        """Create test data in the database."""
        if data_type == "users":
            test_data = [TestDataFactory.create_user() for _ in range(count)]
        elif data_type == "companies":
            test_data = [TestDataFactory.create_company() for _ in range(count)]
        elif data_type == "products":
            test_data = [TestDataFactory.create_product() for _ in range(count)]
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        # Insert into database (mock implementation)
        for item in test_data:
            db_session.add(item)
        db_session.commit()

        return test_data


class APIHelpers:
    """Helper methods for API testing."""

    @staticmethod
    def create_test_client(app):
        """Create a test client for the application."""
        # from fastapi.testclient import TestClient
        # return TestClient(app)
        return None  # Placeholder

    @staticmethod
    def authenticate_client(client, user_credentials: Dict[str, str]):
        """Authenticate a test client and return authorization headers."""
        # response = client.post("/api/auth/login", json=user_credentials)
        # token = response.json()["access_token"]
        # return {"Authorization": f"Bearer {token}"}
        return {"Authorization": "Bearer test_token"}  # Placeholder

    @staticmethod
    def make_authenticated_request(client, method: str, url: str, headers: Dict[str, str], **kwargs):
        """Make an authenticated request."""
        # Combine auth headers with any additional headers
        request_headers = {**headers}
        request_headers.update(kwargs.get("headers", {}))

        # Make the request
        # return getattr(client, method.lower())(url, headers=request_headers, **kwargs)
        return None  # Placeholder


class PerformanceHelpers:
    """Helper methods for performance testing."""

    @staticmethod
    def measure_execution_time(func: Callable, *args, **kwargs) -> tuple:
        """Measure the execution time of a function."""
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time

    @staticmethod
    def calculate_throughput(operations: int, time_seconds: float) -> float:
        """Calculate operations per second."""
        return operations / time_seconds if time_seconds > 0 else 0

    @staticmethod
    def calculate_percentile(values: List[float], percentile: float) -> float:
        """Calculate a percentile from a list of values."""
        import statistics
        return statistics.quantiles(values, n=100)[int(percentile) - 1]

    @staticmethod
    def assert_performance_threshold(actual: float, threshold: float, metric_name: str):
        """Assert that a performance metric meets a threshold."""
        assert actual <= threshold, \
            f"{metric_name} {actual} exceeds threshold {threshold}"


# Convenience instances for easy importing
test_data_factory = TestDataFactory()
mock_factory = MockFactory()
assertion_helpers = AssertionHelpers()
filesystem_helpers = FileSystemHelpers()
database_helpers = DatabaseHelpers()
api_helpers = APIHelpers()
performance_helpers = PerformanceHelpers()