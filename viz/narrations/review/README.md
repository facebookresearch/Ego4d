
# Mephisto Review App

A customizable base template for creating data exploration interfaces with first-class support for the ```mephisto review``` command.

## Table of Contents
1. [Usage](#Usage)
2. [Notes](#Notes)
3. [Customization](#Customization)

## Usage

1. Create a sample data file ```sample-data.csv``` (this template also ships with a sample data file for you):

    ```
    This is good text, row1
    This is bad text, row2
    ```

2. Use `create-react-app` to generate a review app with this template as a base

    ```npx create-react-app my-review --template mephisto-review```

3. Build your app

    cd my-review
    yarn build

4. Run ```mephisto review``` (note that [Mephisto must be installed](https://github.com/facebookresearch/Mephisto/blob/main/docs/quickstart.md) to use this CLI command)

    $ cat sample-data.csv | mephisto review ~/path/to/your/my-review/build --all -o results.csv

5. Open the review app in your browser

If the review CLI command ran correctly, you should see output similar to the following on your terminal:

    Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

Open ```http://127.0.0.1:5000/``` or the given URL in an internet browser

6. Review your sample data with the UI prompts

7. Press ```Ctrl-C``` in your terminal to shut down Mephisto review if it has not shut down automatically

8. View your review results in the generated ```results.csv```

## Notes

- The ID property of review items represents their 0-indexed position within the given list of review items for Mephisto review.
- If you wish to review jsonl type data, remember to add the --json flag to your Mephisto review command for proper data parsing.
- If you wish to review all items in an unordered fashion include the ```--all``` flag in your Mephisto review command
- If you wish to review all items once in an ordered fashion do not include the ```--all``` flag in your Mephisto review command

## Customization

To customize the review interface, make modifications to the ```index.js``` file
You can customize the way review items are displayed for both of the two routes in the App:

1. Customize the layout of all data:

    - **Modify the properties of the ```<CollectionView/>``` tag under the ```/``` route on line 35.**

    - If you do not wish to use pagination, set the ```pagination``` property to ```false``` (default is true, must be a boolean value)

    - Adjust the number of results that appear per page of review by setting the ```resultsPerPage``` property (must be an integer)

    - Add custom renderers for rendering the layout of all items by passing a ```collectionRenderer``` property to AllItemView:
        - The collectionRenderer property must be a react component.
        - The collectionRenderer component will be passed down a property of ```items```, which represents an array of all review items.
        - The collectionRenderer component will also be passed down a property of ```itemRenderer``` which can collectionRenderer the data of a single item.
        - The itemRenderer can be placed in each of the individual item views of your collectionRenderer
        - By default the item collectionRenderer will be a header displaying the item id as well as a pre element containing the stringified JSON data of the item

    - Customize the layout of individual items in either the default collectionRenderer or your custom collectionRenderer by passing an ```itemRenderer``` property to AllItemView:
        - The itemRenderer must be a react component.
        - The itemRenderer will be passed down a property of ```item```
        - The item property will contain the properties of ```items``` representing the JSON data of the review item as well as an ```id``` representing the 0 indexed position of the item in the review data

2. Customize the layout of individual item views:

    - **Modify the properties of the ```<ItemView/>``` tag under the ```/``` route on line 24.**

    - Add custom renderers for items by passing an ```itemRenderer``` property to ItemView:
        - The itemRenderer property must be a react component.
        - The itemRenderer component will be passed down a property of ```item```
        - The item property will contain the properties of ```items``` representing the JSON data of the review item as well as an ```id``` representing the 0 indexed position of the item in the review data
