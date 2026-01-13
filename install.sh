#!/bin/bash
#
# Constat Installation Script
# Installs system dependencies and Python package
#
# Usage:
#   ./install.sh              # Interactive installation
#   ./install.sh --yes        # Non-interactive, install all optional
#   ./install.sh --minimal    # Non-interactive, skip all optional
#
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
header() { echo -e "\n${BOLD}${CYAN}$1${NC}\n"; }

# Parse arguments
INTERACTIVE=true
INSTALL_ALL=false
for arg in "$@"; do
    case $arg in
        --yes|-y)
            INTERACTIVE=false
            INSTALL_ALL=true
            ;;
        --minimal|-m)
            INTERACTIVE=false
            INSTALL_ALL=false
            ;;
    esac
done

# Prompt user (returns 0 for yes, 1 for no)
ask() {
    if [[ "$INTERACTIVE" == "false" ]]; then
        [[ "$INSTALL_ALL" == "true" ]] && return 0 || return 1
    fi
    local prompt="$1"
    local default="${2:-n}"
    local response
    if [[ "$default" == "y" ]]; then
        read -p "$prompt [Y/n]: " response
        [[ -z "$response" || "$response" =~ ^[Yy] ]] && return 0 || return 1
    else
        read -p "$prompt [y/N]: " response
        [[ "$response" =~ ^[Yy] ]] && return 0 || return 1
    fi
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ -f /etc/debian_version ]]; then
        echo "debian"
    elif [[ -f /etc/redhat-release ]]; then
        echo "redhat"
    elif [[ -f /etc/arch-release ]]; then
        echo "arch"
    elif [[ -f /etc/alpine-release ]]; then
        echo "alpine"
    else
        echo "unknown"
    fi
}

# Check if command exists
has_command() {
    command -v "$1" &> /dev/null
}

# Install system dependencies based on OS
install_system_deps() {
    local os=$(detect_os)
    header "System Dependencies"
    info "Detected OS: $os"

    echo ""
    echo "Constat requires graphviz (system package) for graph visualization."
    echo ""

    case $os in
        macos)
            if ! has_command dot; then
                if has_command brew; then
                    info "Installing graphviz via Homebrew..."
                    if brew install graphviz; then
                        success "graphviz installed"
                    else
                        warn "graphviz install had warnings, but may still work"
                    fi
                else
                    warn "Homebrew not found. Install graphviz manually:"
                    echo "  brew install graphviz"
                fi
            else
                success "graphviz already installed"
            fi
            ;;

        debian)
            if ! has_command dot; then
                info "Installing graphviz via apt..."
                sudo apt-get update
                sudo apt-get install -y graphviz libgraphviz-dev
                success "graphviz installed"
            else
                success "graphviz already installed"
            fi
            ;;

        redhat)
            if ! has_command dot; then
                info "Installing graphviz via dnf/yum..."
                if has_command dnf; then
                    sudo dnf install -y graphviz graphviz-devel
                else
                    sudo yum install -y graphviz graphviz-devel
                fi
                success "graphviz installed"
            else
                success "graphviz already installed"
            fi
            ;;

        arch)
            if ! has_command dot; then
                info "Installing graphviz via pacman..."
                sudo pacman -Sy --noconfirm graphviz
                success "graphviz installed"
            else
                success "graphviz already installed"
            fi
            ;;

        alpine)
            if ! has_command dot; then
                info "Installing graphviz via apk..."
                sudo apk add --no-cache graphviz graphviz-dev
                success "graphviz installed"
            else
                success "graphviz already installed"
            fi
            ;;

        *)
            warn "Unknown OS. Please install graphviz manually:"
            echo "  - macOS:        brew install graphviz"
            echo "  - Debian/Ubuntu: sudo apt install graphviz"
            echo "  - Fedora/RHEL:   sudo dnf install graphviz"
            echo "  - Arch:          sudo pacman -S graphviz"
            echo "  - Alpine:        sudo apk add graphviz"
            ;;
    esac
}

# Check Python version
check_python() {
    header "Python Check"

    if ! has_command python3; then
        error "Python 3 is required but not found. Please install Python 3.11+"
    fi

    local version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    local major=$(echo $version | cut -d. -f1)
    local minor=$(echo $version | cut -d. -f2)

    if [[ $major -lt 3 ]] || [[ $major -eq 3 && $minor -lt 11 ]]; then
        error "Python 3.11+ required, found $version"
    fi

    success "Python $version"
}

# Create virtual environment
setup_venv() {
    header "Virtual Environment"

    if [[ ! -d ".venv" ]]; then
        info "Creating virtual environment..."
        python3 -m venv .venv
        success "Created .venv/"
    else
        success "Using existing .venv/"
    fi

    source .venv/bin/activate
    pip install --upgrade pip -q
}

# Show what's included in base install
show_included() {
    header "Core Dependencies (included in base install)"

    echo "These packages are installed automatically:"
    echo ""
    echo "  ${GREEN}Core${NC}"
    echo "    - pydantic, pyyaml, sqlalchemy, pandas, numpy"
    echo "    - anthropic (Claude LLM)"
    echo "    - sentence-transformers (embeddings)"
    echo ""
    echo "  ${GREEN}CLI & REPL${NC}"
    echo "    - rich, click, prompt_toolkit"
    echo ""
    echo "  ${GREEN}Visualization${NC}"
    echo "    - plotly, altair, matplotlib, seaborn, graphviz"
    echo ""
    echo "  ${GREEN}Documents${NC}"
    echo "    - pypdf (PDF files)"
    echo "    - python-docx (Word documents)"
    echo "    - openpyxl (Excel files)"
    echo "    - python-pptx (PowerPoint files)"
    echo ""
    echo "  ${GREEN}Databases${NC}"
    echo "    - PostgreSQL (psycopg2)"
    echo "    - DuckDB"
    echo "    - SQLite (built into Python)"
    echo ""
    echo "  ${GREEN}LLM Providers${NC}"
    echo "    - Anthropic Claude (default)"
    echo "    - Ollama (local models)"
    echo ""
    echo "  ${GREEN}API${NC}"
    echo "    - GraphQL server (strawberry-graphql, fastapi, uvicorn)"
}

# Install base package
install_base() {
    header "Installing Constat"

    info "Installing core package..."
    pip install -e . -q
    success "Core package installed"
}

# Handle optional dependencies
install_optional() {
    header "Optional Dependencies"

    echo "The following are ${YELLOW}optional${NC} and require additional setup:"
    echo ""

    # MySQL
    echo "  ${CYAN}MySQL${NC}"
    echo "    Adds MySQL/MariaDB database support"
    echo "    Install: pip install constat[mysql]"
    echo ""

    # OpenAI
    echo "  ${CYAN}OpenAI${NC}"
    echo "    Adds OpenAI GPT models as alternative to Claude"
    echo "    Install: pip install constat[openai]"
    echo "    Requires: OPENAI_API_KEY environment variable"
    echo ""

    # NoSQL Databases
    echo "  ${CYAN}NoSQL Databases${NC}"
    echo "    MongoDB:       pip install constat[mongodb]"
    echo "    Cassandra:     pip install constat[cassandra]"
    echo "    Elasticsearch: pip install constat[elasticsearch]"
    echo "    DynamoDB:      pip install constat[dynamodb]"
    echo "    Cosmos DB:     pip install constat[cosmosdb]"
    echo "    Firestore:     pip install constat[firestore]"
    echo ""

    # Ask about each optional
    local extras=""

    if ask "Install MySQL support?"; then
        extras="${extras}mysql,"
    fi

    if ask "Install OpenAI support?"; then
        extras="${extras}openai,"
    fi

    if ask "Install NoSQL connectors (MongoDB, Cassandra, etc.)?"; then
        extras="${extras}mongodb,cassandra,elasticsearch,dynamodb,cosmosdb,firestore,"
    fi

    # Install selected extras
    if [[ -n "$extras" ]]; then
        extras="${extras%,}"  # Remove trailing comma
        info "Installing optional dependencies: $extras"
        pip install -e ".[$extras]" -q
        success "Optional dependencies installed"
    else
        info "Skipping optional dependencies"
    fi
}

# Verify installation
verify_installation() {
    header "Verification"

    # Check core imports
    python3 -c "import constat" || error "Core import failed"
    success "Core package: OK"

    # Check key modules
    python3 -c "import plotly" && success "Visualization: OK" || warn "plotly not available"
    python3 -c "import pypdf" && success "PDF support: OK" || warn "pypdf not available"
    python3 -c "import psycopg2" && success "PostgreSQL: OK" || warn "psycopg2 not available"
    python3 -c "import duckdb" && success "DuckDB: OK" || warn "duckdb not available"
    python3 -c "import graphviz" && success "Graphviz: OK" || warn "graphviz not available"
}

# Show next steps
show_next_steps() {
    header "Installation Complete!"

    echo "To get started:"
    echo ""
    echo "  ${CYAN}1. Activate the virtual environment:${NC}"
    echo "     source .venv/bin/activate"
    echo ""
    echo "  ${CYAN}2. Set your API key:${NC}"
    echo "     export ANTHROPIC_API_KEY=your-key-here"
    echo ""
    echo "  ${CYAN}3. Run the demo:${NC}"
    echo "     constat repl -c demo/config.yaml"
    echo ""
    echo "  ${CYAN}4. Or create your own config:${NC}"
    echo "     constat init my-config.yaml"
    echo ""

    if [[ "$INTERACTIVE" == "true" ]]; then
        echo "To install additional dependencies later:"
        echo "  pip install constat[mysql]      # MySQL"
        echo "  pip install constat[mongodb]    # MongoDB"
        echo "  pip install constat[openai]     # OpenAI"
        echo "  pip install constat[all]        # Everything"
    fi
}

# Main
main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                 Constat Installation Script                    ║"
    echo "║         Multi-Step AI Reasoning Engine for Data Analysis       ║"
    echo "╚════════════════════════════════════════════════════════════════╝"

    check_python
    install_system_deps
    setup_venv
    show_included
    install_base
    install_optional
    verify_installation
    show_next_steps
}

main "$@"
