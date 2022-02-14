// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import React from "react";
import { Link } from "react-router-dom";
import { JSONItem } from "../JSONItem";
import "./GridCollection.css";

function GridCollection({ items, itemRenderer: ItemRenderer = JSONItem }) {
  return items && items.length > 0 ? (
    <div className="default-collection-renderer-container">
      {items.map((item) => {
        return (
          <Link
            to={`/${item.id}`}
            style={{ textDecoration: "none" }}
            key={item.id}
            id={`item-${item.id}`}
          >
            <ItemRenderer item={item} />
          </Link>
        );
      })}
    </div>
  ) : null;
}

export { GridCollection };
