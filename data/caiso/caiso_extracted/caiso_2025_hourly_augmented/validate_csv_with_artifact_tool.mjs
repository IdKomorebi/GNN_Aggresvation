import fs from "node:fs/promises";
import { Workbook } from "@oai/artifact-tool";

const [csvPath, sheetName, finalRow, finalColumn] = process.argv.slice(2);
if (!csvPath || !sheetName || !finalRow || !finalColumn) {
  throw new Error(
    "usage: node validate_csv_with_artifact_tool.mjs <csv> <sheet> <finalRow> <finalColumn>",
  );
}

const csvText = await fs.readFile(csvPath, "utf8");
const workbook = await Workbook.fromCSV(csvText, { sheetName });

const overview = await workbook.inspect({
  kind: "workbook,sheet",
  maxChars: 3000,
});
const header = await workbook.inspect({
  kind: "region",
  sheetId: sheetName,
  range: `A1:${finalColumn}4`,
  maxChars: 5000,
  tableMaxRows: 4,
  tableMaxCols: 12,
  tableMaxCellChars: 80,
});
const tail = await workbook.inspect({
  kind: "region",
  sheetId: sheetName,
  range: `A${Number(finalRow) - 2}:H${finalRow}`,
  maxChars: 3000,
  tableMaxRows: 3,
  tableMaxCols: 8,
  tableMaxCellChars: 80,
});

console.log(overview.ndjson);
console.log(header.ndjson);
console.log(tail.ndjson);
