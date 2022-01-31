import React from "react";
import { Button, Icon, Intent } from "@blueprintjs/core";
import "./Pagination.css";

/*
 * Creates pagination buttons based on the total number of pages to be paginated as well as the active page.
 * The setPage parameter allows the component to change the page state in its parent component.
 */

function getPagination(page, totalPages) {
  //how many buttons are allowed to be displayed between the next and last page buttons
  const MAX_CELLS = 7;
  //middle entry of the buttons is the floor of half of the max
  const CELL_MID_LEN = Math.floor(MAX_CELLS / 2);

  /*stores the attributes of the buttons to be displayed
   * The term ellipsis in objects stored in this array refers to series of page numbers condensed into a single "..." button
   * The term number in objects stored in this array refers to the page number of the button if the button is not an ellipsis
   */
  let pages = [];

  if (totalPages > MAX_CELLS) {
    //if there are more pages than the maximum buttons we can display, calculate where to place ellipis

    //start by populating the first and last two buttons in the display with their page numbers
    pages[0] = { number: 1 };
    pages[1] = { number: 2 };
    pages[MAX_CELLS - 2] = { number: totalPages - 1 };
    pages[MAX_CELLS - 1] = { number: totalPages };

    if (page <= CELL_MID_LEN) {
      /*
       * if active page is less than or equal to the middle entry of the display
       * place ellipsis before the last button and populate the cells from the first button on
       */
      pages[MAX_CELLS - 2].ellipsis = true;
      for (let i = 2; i < MAX_CELLS - 2; i++) {
        pages[i] = { number: i + 1 };
      }
    } else if (totalPages - page < CELL_MID_LEN) {
      /*
       * else if the distance from active page to the last page is less than the middle entry of the display
       * place ellipsis after the first button and populate the cells from the last button prior
       */
      pages[1].ellipsis = true;
      for (let i = 2; i < MAX_CELLS - 2; i++) {
        pages[i] = { number: totalPages - MAX_CELLS + i + 1 };
      }
    } else {
      /*
       * otherwise the active page must be too far away from first and last buttons and two ellipsis must be placed on either side of the active pages
       * cells are populated assuming the middle button is the active page
       */
      pages[1].ellipsis = true;
      pages[MAX_CELLS - 2].ellipsis = true;

      pages[CELL_MID_LEN] = { number: page };
      for (let i = 1; i < MAX_CELLS - 5; i++) {
        pages[CELL_MID_LEN + i] = { number: page + i };
        pages[CELL_MID_LEN - i] = { number: page - i };
      }
    }
  } else {
    //If there are less pages than the max no ellipsis is needed and all buttons can be populated normally
    for (let i = 0; i < totalPages; i++) {
      pages[i] = { number: i + 1, ellipsis: false };
    }
  }

  //set active flag for active page
  pages.forEach((p) => {
    if (p.number === page) p.active = true;
  });

  //if at last page disable going forward and if at first page disable going back
  const isLeftArrowEnabled = page > 1;
  const isRightArrowEnabled = page < totalPages;

  return {
    pages,
    isLeftArrowEnabled,
    isRightArrowEnabled,
  };
}

function Pagination({ page = 1, totalPages = 1, setPage = () => {} }) {
  const { pages, isLeftArrowEnabled, isRightArrowEnabled } = getPagination(
    page,
    totalPages
  );

  return (
    <div className="bp3-button-group pagination">
      <Button
        large={true}
        disabled={!isLeftArrowEnabled}
        onClick={() => setPage(page - 1)}
        id="pagination-button-left"
      >
        <Icon icon="chevron-left" />
      </Button>
      {pages.map((p) => (
        <Button
          large={true}
          text={p.ellipsis ? "..." : p.number}
          key={p.number}
          disabled={p.ellipsis}
          intent={p.active ? Intent.PRIMARY : Intent.DEFAULT}
          onClick={() => setPage(p.number)}
        />
      ))}
      <Button
        large={true}
        disabled={!isRightArrowEnabled}
        onClick={() => setPage(page + 1)}
        id="pagination-button-right"
      >
        <Icon icon="chevron-right" />
      </Button>
    </div>
  );
}

export default Pagination;
