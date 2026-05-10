# Copyright (c) 2025 Kenneth Stott
# Canary: aeb1c9f1-2789-43b7-90a1-3ab306d63720
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Strawberry GraphQL type definitions — re-exports from per-domain modules.

All existing ``from constat.server.graphql.types import X`` imports continue
to work unchanged.  New types should be added to the appropriate sub-module.
"""

# -- Glossary types --
from constat.server.graphql.types.glossary_types import (  # noqa: F401
    ConnectedResourceSource,
    ConnectedResourceType,
    DraftAliasesType,
    DraftDefinitionType,
    DraftTagsType,
    EntityRelationshipType,
    GenerateResultType,
    GlossaryChangeAction,
    GlossaryChangeEvent,
    GlossaryChildType,
    GlossaryListType,
    GlossaryParentType,
    GlossaryTermInput,
    GlossaryTermType,
    GlossaryTermUpdateInput,
    RefineResultType,
    RenameResultType,
    TaxonomySuggestionType,
    TaxonomySuggestionsType,
)

# -- Auth types --
from constat.server.graphql.types.auth_types import (  # noqa: F401
    AuthPayload,
    ModelRouteInfoType,
    PasskeyOptions,
    ServerConfigType,
    UserPermissionsType,
)

# -- Session types --
from constat.server.graphql.types.session_types import (  # noqa: F401
    SessionListType,
    SessionStatusEnum,
    SessionType,
    ShareResponseType,
    TogglePublicResponseType,
)

# -- State types (Phase 3) --
from constat.server.graphql.types.state_types import (  # noqa: F401
    ActiveAgentType,
    ActiveSkillType,
    ApiEndpointType,
    ApiFieldType,
    ApiSchemaType,
    DatabaseSchemaType,
    DatabaseTableInfoType,
    ExecutionOutputType,
    InferenceCodeListType,
    InferenceCodeType,
    MessagesType,
    ObjectivesEntryType,
    PromptContextType,
    ProofFactsType,
    ProofTreeNodeType,
    ProofTreeType,
    SaveResultType,
    ScratchpadEntryType,
    ScratchpadType,
    StepCodeListType,
    StepCodeType,
    StoredMessageType,
    StoredProofFactType,
    UpdateSystemPromptResultType,
    PublicSessionSummaryType,
)

# -- Data types (Phase 4) --
from constat.server.graphql.types.data_types import (  # noqa: F401
    AddEntityToGlossaryResultType,
    ArtifactContentType,
    ArtifactInfoType,
    ArtifactListType,
    ArtifactVersionInfoType,
    ArtifactVersionsType,
    DeleteResultType,
    EntityInfoType,
    EntityListType,
    EntityReferenceInfoType,
    FactInfoType,
    FactListType,
    FactMutationResultType,
    MoveFactResultType,
    TableDataType,
    TableInfoType,
    TableListType,
    TableVersionInfoType,
    TableVersionsType,
    ToggleStarResultType,
)

# -- Source types (Phase 5) --
from constat.server.graphql.types.source_types import (  # noqa: F401
    ApiAddInput,
    ApiUpdateInput,
    DatabaseAddInput,
    DatabaseUpdateInput,
    DatabaseTablePreviewType,
    DatabaseTestResultType,
    DataSourcesType,
    DocumentResultType,
    DocumentUriInput,
    DocumentUpdateInput,
    EmailSourceInput,
    FileRefInput,
    FileRefListType,
    FileRefType,
    MoveSourceResultType,
    SessionApiType,
    SessionDatabaseListType,
    SessionDatabaseType,
    SessionDocumentType,
    UploadDocumentResultItem,
    UploadDocumentsResultType,
    UploadedFileListType,
    UploadedFileType,
    UserSourceResultType,
    UserSourcesType,
)

# -- Execution types (Phase 7) --
from constat.server.graphql.types.execution_types import (  # noqa: F401
    ApprovePlanInput,
    AutocompleteItemType,
    AutocompleteResultType,
    EditedStepInput,
    ExecutionActionResultType,
    ExecutionEventType,
    ExecutionPlanType,
    PlanStepType,
    QuerySubmissionType,
    SubmitQueryInput,
)

# -- Learning types (Phase 8) --
from constat.server.graphql.types.learning_types import (  # noqa: F401
    AgentContentType,
    AgentInfoType,
    CompactionResultType,
    CreateAgentInput,
    CreateRuleInput,
    CreateSkillFromProofInput,
    CreateSkillFromProofResultType,
    CreateSkillInput,
    DraftAgentInput,
    DraftAgentResultType,
    DraftSkillInput,
    DraftSkillResultType,
    LearningInfoType,
    LearningListType,
    RuleInfoType,
    SetActiveSkillsResultType,
    SetAgentResultType,
    SkillContentType,
    SkillInfoType,
    SkillsListType,
    UpdateAgentInput,
    UpdateRuleInput,
    UpdateSkillInput,
)

# -- Secondary types (Phase 9): fine-tune, feedback, testing, OAuth --
from constat.server.graphql.types.secondary_types import (  # noqa: F401
    CreateGoldenQuestionInput,
    EmailOAuthProvidersType,
    FineTuneJobType,
    FineTuneProviderType,
    FlagAnswerInput,
    FlagAnswerResultType,
    GlossarySuggestionType,
    GoldenQuestionExpectInput,
    GoldenQuestionExpectationType,
    GoldenQuestionType,
    MoveGoldenQuestionInput,
    StartFineTuneInput,
    SuggestionActionResultType,
    TestableDomainType,
    UpdateGoldenQuestionInput,
)

# -- Domain types (Phase 6) --
from constat.server.graphql.types.domain_types import (  # noqa: F401
    CreateDomainInput,
    CreateDomainResultType,
    DeleteDomainResultType,
    DomainAgentType,
    DomainContentSaveResultType,
    DomainContentType,
    DomainDetailType,
    DomainFactType,
    DomainInfoType,
    DomainListType,
    DomainRuleType,
    DomainSkillType,
    DomainTreeNodeType,
    MoveDomainAgentInput,
    MoveDomainAgentResultType,
    MoveDomainRuleInput,
    MoveDomainRuleResultType,
    MoveDomainSkillInput,
    MoveDomainSkillResultType,
    MoveDomainSourceInput,
    MoveDomainSourceResultType,
    PromoteDomainResultType,
    UpdateDomainInput,
    UpdateDomainResultType,
)
