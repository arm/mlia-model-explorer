// SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0

export type SourceLine = {
  path: string;
  line: number;
  column?: number;
};

export type MappingEntry = {
  model_explorer_node_id: string;
  source?: SourceLine[];
};

export type MappingData = {
  mappings: MappingEntry[];
};

export function getSourceCodeLinesForModelExplorerNode(
  modelExplorerNodeId: string,
  mappingData: MappingData,
): SourceLine[] {
  return mappingData.mappings
    .filter((entry) => entry.model_explorer_node_id === modelExplorerNodeId)
    .flatMap((entry) => entry.source ?? []);
}

export function getModelExplorerNodesForSourceLine(
  sourceCodeLine: SourceLine,
  mappingData: MappingData,
): string[] {
  return mappingData.mappings
    .filter((entry) =>
      (entry.source ?? []).some(
        (source) => source.path === sourceCodeLine.path && source.line === sourceCodeLine.line,
      ),
    )
    .map((entry) => entry.model_explorer_node_id);
}

export function configureModelExplorerHooks(
  modelExplorerElement: EventTarget,
  mappingData: MappingData,
  sourceCodeHighlightCallback: (sourceLines: SourceLine[]) => void,
): (sourceCodeLine: SourceLine) => string[] {
  modelExplorerElement.addEventListener("node-selected", (event: Event) => {
    const customEvent = event as CustomEvent<{ nodeId: string }>;
    const nodeId = customEvent.detail?.nodeId;
    if (nodeId) {
      sourceCodeHighlightCallback(
        getSourceCodeLinesForModelExplorerNode(nodeId, mappingData),
      );
    }
  });

  return (sourceCodeLine: SourceLine) =>
    getModelExplorerNodesForSourceLine(sourceCodeLine, mappingData);
}
