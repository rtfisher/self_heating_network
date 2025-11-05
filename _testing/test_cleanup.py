import pytest
import os
import tempfile
import shutil
from cleanup import clean_up_specific_files

# Test functions for cleanup.py module

@pytest.mark.unit
class TestCleanupFunctions:
    """Test suite for cleanup.py file removal functions"""

    @pytest.fixture
    def temp_directory(self):
        """Create a temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir)

    def test_cleanup_abundances_file(self, temp_directory):
        """Test cleanup of abundances.png file"""
        # Create test file
        test_file = os.path.join(temp_directory, "abundances.png")
        with open(test_file, 'w') as f:
            f.write("test")

        assert os.path.exists(test_file), "Test file should exist before cleanup"

        # Run cleanup
        clean_up_specific_files(temp_directory)

        assert not os.path.exists(test_file), "abundances.png should be removed"

    def test_cleanup_detonation_lengths_file(self, temp_directory):
        """Test cleanup of detonation_lengths.png file"""
        # Create test file
        test_file = os.path.join(temp_directory, "detonation_lengths.png")
        with open(test_file, 'w') as f:
            f.write("test")

        assert os.path.exists(test_file), "Test file should exist before cleanup"

        # Run cleanup
        clean_up_specific_files(temp_directory)

        assert not os.path.exists(test_file), "detonation_lengths.png should be removed"

    def test_cleanup_reaction_flow_pattern(self, temp_directory):
        """Test cleanup of reaction_flow_*.png pattern files"""
        # Create multiple test files matching pattern
        test_files = [
            os.path.join(temp_directory, "reaction_flow_0.00.png"),
            os.path.join(temp_directory, "reaction_flow_0.10.png"),
            os.path.join(temp_directory, "reaction_flow_1.50.png")
        ]

        for test_file in test_files:
            with open(test_file, 'w') as f:
                f.write("test")
            assert os.path.exists(test_file), f"{test_file} should exist before cleanup"

        # Run cleanup
        clean_up_specific_files(temp_directory)

        # Check all files are removed
        for test_file in test_files:
            assert not os.path.exists(test_file), f"{test_file} should be removed"

    def test_cleanup_preserves_other_files(self, temp_directory):
        """Test that cleanup preserves files not in the cleanup list"""
        # Create files that should NOT be deleted
        preserved_files = [
            os.path.join(temp_directory, "self_heat.py"),
            os.path.join(temp_directory, "aux.py"),
            os.path.join(temp_directory, "test_data.txt"),
            os.path.join(temp_directory, "other_plot.png")
        ]

        for test_file in preserved_files:
            with open(test_file, 'w') as f:
                f.write("test")
            assert os.path.exists(test_file), f"{test_file} should exist before cleanup"

        # Run cleanup
        clean_up_specific_files(temp_directory)

        # Check all files still exist
        for test_file in preserved_files:
            assert os.path.exists(test_file), f"{test_file} should be preserved"

    def test_cleanup_empty_directory(self, temp_directory):
        """Test cleanup on empty directory doesn't raise errors"""
        # Run cleanup on empty directory (should not raise any exceptions)
        try:
            clean_up_specific_files(temp_directory)
        except Exception as e:
            pytest.fail(f"Cleanup raised an exception on empty directory: {e}")

    def test_cleanup_mixed_files(self, temp_directory):
        """Test cleanup with mix of target and non-target files"""
        # Create target files
        target_files = [
            os.path.join(temp_directory, "abundances.png"),
            os.path.join(temp_directory, "detonation_lengths.png"),
            os.path.join(temp_directory, "reaction_flow_5.23.png")
        ]

        # Create non-target files
        preserved_files = [
            os.path.join(temp_directory, "important_data.dat"),
            os.path.join(temp_directory, "config.txt")
        ]

        for test_file in target_files + preserved_files:
            with open(test_file, 'w') as f:
                f.write("test")

        # Run cleanup
        clean_up_specific_files(temp_directory)

        # Check target files removed
        for test_file in target_files:
            assert not os.path.exists(test_file), f"{test_file} should be removed"

        # Check preserved files still exist
        for test_file in preserved_files:
            assert os.path.exists(test_file), f"{test_file} should be preserved"

    def test_cleanup_nonexistent_files(self, temp_directory):
        """Test cleanup when target files don't exist (should not raise errors)"""
        # Don't create any files, just run cleanup
        try:
            clean_up_specific_files(temp_directory)
        except Exception as e:
            pytest.fail(f"Cleanup raised an exception when files don't exist: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
