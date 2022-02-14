// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import React from "react";
import { Button, Icon } from "@blueprintjs/core";

function ErrorPane({ error }) {
  const [hide, setHide] = React.useState(false);
  const errorType = error && error.type;
  React.useEffect(() => {
    if (!error) return;
    console.error(error);
  }, [error, errorType]);
  return (
    error &&
    !hide && (
      <div className="error item-view-error">
        <Button
          icon={<Icon icon="cross" color="white" />}
          minimal
          onClick={() => setHide(true)}
        />
        Error [{error.type}] &mdash; {JSON.stringify(error.error)}
      </div>
    )
  );
}

export default ErrorPane;
