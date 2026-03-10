import { describe, expect, it, vi } from "vitest";
import {
  formatHelpExample,
  formatHelpExampleGroup,
  formatHelpExampleLine,
  formatHelpExamples,
} from "./help-format.js";

vi.mock("../terminal/theme.js", () => ({
  theme: {
    command: (s: string) => `COMMAND(${s})`,
    muted: (s: string) => `MUTED(${s})`,
  },
}));

describe("help-format", () => {
  describe("formatHelpExample", () => {
    it("formats a command and description on two lines", () => {
      const result = formatHelpExample("my-command", "does something");
      expect(result).toBe("  COMMAND(my-command)\n    MUTED(does something)");
    });
  });

  describe("formatHelpExampleLine", () => {
    it("formats a command and description on one line", () => {
      const result = formatHelpExampleLine("my-command", "does something");
      expect(result).toBe("  COMMAND(my-command) MUTED(# does something)");
    });

    it("formats only the command if description is missing", () => {
      const result = formatHelpExampleLine("my-command", "");
      expect(result).toBe("  COMMAND(my-command)");
    });
  });

  describe("formatHelpExamples", () => {
    const examples = [
      ["cmd1", "desc1"],
      ["cmd2", "desc2"],
    ] as const;

    it("formats multiple examples (default)", () => {
      const result = formatHelpExamples(examples);
      expect(result).toBe(
        "  COMMAND(cmd1)\n    MUTED(desc1)\n  COMMAND(cmd2)\n    MUTED(desc2)",
      );
    });

    it("formats multiple examples inline", () => {
      const result = formatHelpExamples(examples, true);
      expect(result).toBe(
        "  COMMAND(cmd1) MUTED(# desc1)\n  COMMAND(cmd2) MUTED(# desc2)",
      );
    });
  });

  describe("formatHelpExampleGroup", () => {
    const examples = [["cmd1", "desc1"]] as const;

    it("formats a labeled group of examples", () => {
      const result = formatHelpExampleGroup("My Group", examples);
      expect(result).toBe("MUTED(My Group)\n  COMMAND(cmd1)\n    MUTED(desc1)");
    });

    it("formats a labeled group of examples inline", () => {
      const result = formatHelpExampleGroup("My Group", examples, true);
      expect(result).toBe("MUTED(My Group)\n  COMMAND(cmd1) MUTED(# desc1)");
    });
  });
});
