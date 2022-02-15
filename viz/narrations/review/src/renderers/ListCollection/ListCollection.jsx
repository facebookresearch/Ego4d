// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import React from "react";
import { Link } from "react-router-dom";
import { Card, Divider } from "@blueprintjs/core";
import ListItem from "./ListItem";
import "./ListCollection.css";

/*
  EXAMPLE PLUGIN ALL DATA RENDERER
  Displays all mephisto review data as a list
*/
function ListCollection({ items, itemRenderer: Renderer = ListItem }) {
  return items && items.length > 0 ? (
    <Card className="list-view-renderer-container">
      {items.map((item, index) => (
        <>
          {index !== 0 ? <Divider /> : null}
          <Link
            to={`/${item.id}`}
            style={{ textDecoration: "none" }}
            key={item.id}
          >
            <div
              className={
                index !== 0
                  ? "list-view-renderer-item divider"
                  : "list-view-renderer-item"
              }
            >
              <Renderer item={item} />
            </div>
          </Link>
        </>
      ))}
    </Card>
  ) : null;
}

export default ListCollection;
