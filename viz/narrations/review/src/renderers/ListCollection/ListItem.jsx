// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import React from "react";
import { H6 } from "@blueprintjs/core";

function ListItem({ item }) {
  return (
    <>
      <pre>{JSON.stringify(item && item.data)}</pre>
      <H6>
        <b>ID: {item && item.id}</b>
      </H6>
    </>
  );
}

export default ListItem;
