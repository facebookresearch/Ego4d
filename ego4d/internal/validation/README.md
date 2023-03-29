# Validation

This section of the repository validates input metadata files and associated
data files located on S3 or on a local filesystem.

## Error Classification

There are two main classes of errors: warnings and errors (similar to a compiler)
- Errors must *be fixed* before they can be ingested
- Warnings are flagged such that you are aware of them

## Usage
```
python ego4d/internal/validation/cli.py -i "<input_dir>"
```

- By default errors will be logged to S3. You can override this by providing an
output directory via `--output` or `-o`
- `<input_dir>` can be a folder on S3 or on the local filesystem
    - If it is on the local filesystem, you must give a university name such
      that we can dump the errors to S3
- The metadata CSV files can reference local files, doing so will result in a
  set of errors. Such that you do not upload these files to S3, but instead fix them prior to doing so.

## Example
```
python ego4d/internal/validation/cli.py -i s3://ego4d-penn/egoexo/metadata_v1
python ego4d/internal/validation/cli.py -i s3://ego4d-utokyo/egoexo/metadata_v1
```

## Debugging

```
ipython --pdb ego4d/internal/validation/cli.py -- -i s3://ego4d-utokyo/egoexo/metadata_v1
```
