// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import React from "react";
import ReactDOM from "react-dom";
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";
import CollectionView from "./components/CollectionView";
import ItemView from "./components/ItemView";
import "normalize.css/normalize.css";
import "@blueprintjs/icons/lib/css/blueprint-icons.css";
import "@blueprintjs/core/lib/css/blueprint.css";
import "./index.css";

import { GridCollection, JSONItem, WordCloudItem } from "./renderers";
import NarrationsThumbnail from "./custom/NarrationsThumbnail";
import NarrationsItem from "./custom/NarrationsItem";

ReactDOM.render(
  <React.StrictMode>
    <Router>
      <Switch>
        <Route path="/:id">
          {/* For more information see the 'Customization' section of the README.md file. */}
          {/* <ItemView wrapClass="item-dynamic" itemRenderer={JSONItem} /> */}
          <ItemView itemRenderer={NarrationsItem} allowReview={false} />
        </Route>
        <Route path="/">
          {/* For more information see the 'Customization' section of the README.md file. */}
          <CollectionView
            collectionRenderer={GridCollection}
            // itemRenderer={JSONItem}
            itemRenderer={NarrationsThumbnail}
            pagination={true}
            resultsPerPage={12}
          />
        </Route>
      </Switch>
    </Router>
  </React.StrictMode>,
  document.getElementById("root")
);
