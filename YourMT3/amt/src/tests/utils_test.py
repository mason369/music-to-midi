import os
import unittest
from unittest.mock import patch
from io import BytesIO
from utils.utils import download_and_extract
from utils.utils import get_checksum
from utils.utils import merge_file_lists
from utils.utils import reindex_file_list_keys
from utils.utils import remove_ids_from_file_list
from utils.utils import deduplicate_splits


class TestMergeFileListFunctions(unittest.TestCase):

    def test_merge_file_lists(self):
        # Define some example input dictionaries
        file_list_1 = {1: 'file1.txt', 2: 'file2.txt'}
        file_list_2 = {3: 'file3.txt', 4: 'file4.txt'}
        file_list_3 = {5: 'file5.txt', 6: 'file6.txt'}

        # Call the merge_file_lists function with the example input
        merged_file_list = merge_file_lists([file_list_1, file_list_2, file_list_3])

        # Check that the merged dictionary has the correct length and keys/values
        self.assertEqual(len(merged_file_list), 6)
        self.assertEqual(merged_file_list[0], 'file1.txt')
        self.assertEqual(merged_file_list[1], 'file2.txt')
        self.assertEqual(merged_file_list[2], 'file3.txt')
        self.assertEqual(merged_file_list[3], 'file4.txt')
        self.assertEqual(merged_file_list[4], 'file5.txt')
        self.assertEqual(merged_file_list[5], 'file6.txt')

    def test_reindex_file_list_keys(self):
        file_list = {'a': {'id': 1, 'name': 'file1'}, 'b': {'id': 2, 'name': 'file2'}}
        expected_reindexed = {0: {'id': 1, 'name': 'file1'}, 1: {'id': 2, 'name': 'file2'}}
        reindexed = reindex_file_list_keys(file_list)
        self.assertEqual(reindexed, expected_reindexed)

    def test_remove_ids_from_file_list(self):
        file_list = {
            'a': {
                'music_id': 123,
                'name': 'file1'
            },
            'b': {
                'music_id': 222,
                'name': 'file2'
            }
        }
        selected_ids = [123]
        expected_filtered = {0: {'music_id': 222, 'name': 'file2'}}
        filtered = remove_ids_from_file_list(file_list, selected_ids, reindex=True)
        self.assertEqual(filtered, expected_filtered)


class TestGetChecksum(unittest.TestCase):

    def test_get_checksum_z(self):
        # Create a temporary file with some content
        file_name = "temp_test_file.txt"
        with open(file_name, "w") as f:
            f.write("This is a test file")

        # Calculate the expected checksum using an online md5 calculator or a known md5 value
        expected_checksum = "0b26e313ed4a7ca6904b0e9369e5b957"

        # Call the get_checksum function
        calculated_checksum = get_checksum(file_name)

        # Compare the expected and calculated checksums
        self.assertEqual(expected_checksum, calculated_checksum)

        # Clean up the temporary file
        os.remove(file_name)


class TestDeduplicateSplits(unittest.TestCase):

    def test_deduplicate_splits(self):
        # Create sample file lists for splits A and B
        file_list_a = {
            'split1': {
                'some_id': 1,
                'file_name': 'a.jpg'
            },
            'split2': {
                'some_id': 2,
                'file_name': 'b.jpg'
            },
            'split3': {
                'some_id': 3,
                'file_name': 'c.jpg'
            }
        }
        file_list_b = {
            'split4': {
                'some_id': 2,
                'file_name': 'd.jpg'
            },
            'split5': {
                'some_id': 3,
                'file_name': 'e.jpg'
            },
            'split6': {
                'some_id': 6,
                'file_name': 'f.jpg'
            }
        }

        # Remove duplicates between split A and split B
        filtered_file_list_a = deduplicate_splits(file_list_a, file_list_b, reindex=False)

        # Check that the correct IDs have been removed from split A
        expected_file_list_a = {
            'split1': {
                'some_id': 1,
                'file_name': 'a.jpg'
            },
        }
        self.assertDictEqual(filtered_file_list_a, expected_file_list_a)


if __name__ == '__main__':
    unittest.main()